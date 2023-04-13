import numpy as np
from pathlib import Path
from PIL import Image
import imageio
import cv2 as cv
import pickle as pkl
import itertools
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import resnet_cifar10_v2

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 128
EPOCHS = 3
CROP_TO = 32
SEED = 26

PROJECTION_DIM = 2048
LATENT_DIM = 512
WEIGHT_DECAY = 0.0005

NUM_ENCODER_LAYERS = 2
DEPTH = NUM_ENCODER_LAYERS * 9 + 2
NUM_BLOCKS = ((DEPTH - 2) // 9) - 1


def get_encoder():
	# input and backbone
	inputs = layers.Input((CROP_TO, CROP_TO, 3))
	x = layers.Rescaling(scale=1.0 / 127.5, offset=-1)(
		inputs
	)
	x = resnet_cifar10_v2.stem(x)
	x = resnet_cifar10_v2.learner(x, NUM_BLOCKS)
	x = layers.GlobalAveragePooling2D(name="backbone_pool")(x)

	# projection head
	x = layers.Dense(
		PROJECTION_DIM, use_bias=False, kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
	)(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)
	x = layers.Dense(
		PROJECTION_DIM, use_bias=False, kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
	)(x)
	outputs = layers.BatchNormalization()(x)
	return tf.keras.Model(inputs, outputs, name="encoder")

def get_predictor():
	model = tf.keras.Sequential(
		[
			# autoencoder-like structure
			layers.Input((PROJECTION_DIM,)),
			layers.Dense(
				LATENT_DIM,
				use_bias=False,
				kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
			),
			layers.ReLU(),
			layers.BatchNormalization(),
			layers.Dense(PROJECTION_DIM),
		],
		name="predictor",
	)
	return model

def triplet_loss(a, p, n, margin):
	a = tf.math.l2_normalize(a, axis=1)
	p = tf.math.l2_normalize(p, axis=1)
	n = tf.math.l2_normalize(n, axis=1)

	pos_sim = tf.reduce_sum(p * a, axis=1)
	neg_sim = tf.reduce_sum(p * n, axis=1)
	
	loss = tf.math.maximum(0., margin + neg_sim - pos_sim)
	return tf.reduce_mean(loss)


class TripletSiam(tf.keras.Model):
	def __init__(self, encoder, predictor, margin=0.2):
		super().__init__()
		self.encoder = encoder
		self.predictor = predictor
		self.margin = margin
		self.loss_tracker = tf.keras.metrics.Mean(name="loss")

	@property
	def metrics(self):
		return [self.loss_tracker]

	def train_step(self, data):
		ds_anchor, ds_positive, ds_negative = data

		with tf.GradientTape() as tape:
			z_a = self.encoder(ds_anchor)
			z_p = self.encoder(ds_positive)
			z_n = self.encoder(ds_negative)

			p_a = self.predictor(z_a)
			p_p = self.predictor(z_p)
			p_n = self.predictor(z_n)

			loss = triplet_loss(z_a, z_p, z_n, self.margin)

		learnable_params = (
			self.encoder.trainable_variables + self.predictor.trainable_variables
		)
		gradients = tape.gradient(loss, learnable_params)
		self.optimizer.apply_gradients(zip(gradients, learnable_params))
		self.loss_tracker.update_state(loss)
		return {"loss": self.loss_tracker.result()}


class TripletImageGenerator(tf.keras.utils.Sequence):
	def __init__(self, image_triplets, video_dir, batch_size=32, shuffle=True):
		self.image_triplets = image_triplets
		self.video_dir = video_dir
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.indexes = np.arange(len(self.image_triplets))
		self.on_epoch_end()

	def __len__(self):
		return int(np.ceil(len(self.image_triplets) / float(self.batch_size)))

	def __getitem__(self, index):
		indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
		batch = [self.image_triplets[k] for k in indexes]

		anchor_images, positive_images, negative_images = self.__data_generation(batch)
		return [anchor_images, positive_images, negative_images]

	def on_epoch_end(self):
		if self.shuffle:
			np.random.shuffle(self.indexes)

	def __data_generation(self, batch):
		anchor_images = []
		positive_images = []
		negative_images = []

		for triplet in batch:
			anchor, positive, negative = triplet

			anchor_frame = self.__read_video_frame(anchor)
			positive_frame = self.__read_video_frame(positive)
			negative_frame = self.__read_video_frame(negative)

			anchor_resized = np.array(Image.fromarray(anchor_frame).resize((48, 48))) # 10x reduction
			positive_resized = np.array(Image.fromarray(positive_frame).resize((48, 48)))
			negative_resized = np.array(Image.fromarray(negative_frame).resize((48, 48)))

			anchor_images.append(anchor_resized)
			positive_images.append(positive_resized)
			negative_images.append(negative_resized)

		return (
			np.array(anchor_images, dtype=np.float32),
			np.array(positive_images, dtype=np.float32),
			np.array(negative_images, dtype=np.float32),
		)

	def __read_video_frame(self, index):
		video_id, frame_id = index
		images = f"{self.video_dir}/{video_id}.mp4"
		reader = imageio.get_reader(images)

		frame = reader.get_data(frame_id)
		reader.close()

		return frame


with open(str(Path("data/compiled/df_unlabeled_exploded_fil480_230411.pkl")), "rb") as f:
	df_unlabeled_exploded = pkl.load(f)

def create_triplets(df, anchor_idx: Tuple[int, int], num_examples=10, threshold=0.1):
	"""sample from all video frames based on anchor,
	get positive and negative example indices using similarity threshold given normalized trajectory"""
	anchor = df.loc[anchor_idx, "traj_norm"]
	# boolean mask of values close to vs far away from anchor
	mask_pos = (df['traj_norm'] < (anchor + threshold)) & (df['traj_norm'] > (anchor - threshold))
	mask_neg = ~(mask_pos)
	df_subset_pos = df.loc[mask_pos, :]
	df_subset_neg = df.loc[mask_neg, :]
	pos = np.random.choice(df_subset_pos.index, num_examples)
	neg = np.random.choice(df_subset_neg.index, num_examples)
	return list(zip(itertools.repeat(anchor_idx, num_examples), pos, neg))


num_anchors = 2
num_triplets_per_anchor = 2
triplets = []

for anchi in range(num_anchors):
	# Randomly select an anchor image
	anchor_idx = np.random.randint(0, len(df_unlabeled_exploded) - 1)
	anchor = df_unlabeled_exploded.index[anchor_idx]
	triplets_anchi = create_triplets(df_unlabeled_exploded, anchor, num_examples=num_triplets_per_anchor)
	triplets.extend(triplets_anchi)


def load_TS_weights(
	margin=0.2,
	weights_path=str(Path("models/230328_1000triplets/TS_model_weights.h5"))
):
	encoder = get_encoder()
	predictor = get_predictor()
	TS_model = TripletSiam(encoder, predictor, margin)
	optimizer = Adam(learning_rate=0.0001)
	TS_model.compile(optimizer)

	video_directory = str(Path("data/Fitness-AQA/Fitness-AQA_dataset_release_002/Squat/Unlabeled_Dataset/videos"))
	batch_size = 1
	shuffle = True
	train_generator = TripletImageGenerator(triplets, video_directory, batch_size, shuffle)

	epochs = 1
	TS_model.fit(train_generator, epochs=epochs)
	TS_model.load_weights(weights_path)

	return TS_model

def TS_inference(
	images: np.array,
	model: TripletSiam = load_TS_weights(),
	clf_path: str = str(Path("models/nn_model_230328")),
	resize_dim: Tuple[int, int] = (32, 32)
)-> np.ndarray:
	"""
	Extracts embeddings from a video file and makes predictions using a classifier model.
	Args:
		images: video as np.array
		model: TripletSiam model used to extract embeddings.
		clf_path: Path to the trained classifier model.
		resize_dim: Tuple of ints representing the desired dimensions to resize frames to
   Returns:
		Predicted labels as a numpy array, e.g. binary vector of errors (1) or not error (0)
	"""
	embeddings = []
	for frame in images:
		frame_rsz = np.expand_dims(cv.resize(frame, resize_dim), axis=0)
		frame_rsz = tf.convert_to_tensor(frame_rsz, dtype=tf.float32)
		embedding = model.encoder.predict(frame_rsz)
		embeddings.append(embedding.flatten())
	embeddings = np.array(embeddings)
	clf_model = load_model(clf_path)
	probs = clf_model.predict(embeddings)
	predictions = (probs > 0.5).astype(int)
	return predictions
