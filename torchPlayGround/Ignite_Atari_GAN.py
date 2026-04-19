import random
import argparse
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import gymnasium
import numpy as np
from ignite.engine import Engine,Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger

# Logger
log = gymnasium.logger
log.set_level(gymnasium.logger.INFO)

# Hyperparameters
LATENT_VECTOR_SIZE = 100
DISCR_FILTERS = 64
GENER_FILTERS = 64
BATCH_SIZE = 16
IMAGE_SIZE = 64
LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000

# ---------------- Input Wrapper ----------------
class InputWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.observation_space, gymnasium.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gymnasium.spaces.Box(
            self.observation(old_space.low),
            self.observation(old_space.high),
            dtype=np.float32
        )

    def observation(self, observation):
        new_obs = cv2.resize(observation, (IMAGE_SIZE, IMAGE_SIZE))
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32)

# ---------------- Discriminator ----------------
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(input_shape[0], DISCR_FILTERS, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(DISCR_FILTERS, DISCR_FILTERS*2, 4, 2, 1),
            nn.BatchNorm2d(DISCR_FILTERS*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(DISCR_FILTERS*2, DISCR_FILTERS*4, 4, 2, 1),
            nn.BatchNorm2d(DISCR_FILTERS*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(DISCR_FILTERS*4, DISCR_FILTERS*8, 4, 2, 1),
            nn.BatchNorm2d(DISCR_FILTERS*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(DISCR_FILTERS*8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_pipe(x)
        return conv_out.view(-1, 1).squeeze(1)

# ---------------- Generator ----------------
class Generator(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(LATENT_VECTOR_SIZE, GENER_FILTERS*8, 4, 1, 0),
            nn.BatchNorm2d(GENER_FILTERS*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(GENER_FILTERS*8, GENER_FILTERS*4, 4, 2, 1),
            nn.BatchNorm2d(GENER_FILTERS*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(GENER_FILTERS*4, GENER_FILTERS*2, 4, 2, 1),
            nn.BatchNorm2d(GENER_FILTERS*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(GENER_FILTERS*2, GENER_FILTERS, 4, 2, 1),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(True),

            nn.ConvTranspose2d(GENER_FILTERS, output_shape[0], 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pipe(x)

# ---------------- Batch Iterator ----------------
def iterate_batches(envs, batch_size=BATCH_SIZE):
    batch = []
    env_gen = iter(lambda: random.choice(envs), None)

    while True:
        env = next(env_gen)
        obs, _ = env.reset()
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        batch.append(obs)

        if len(batch) == batch_size:
            batch_np = np.stack(batch, axis=0).astype(np.float32)
            batch_np = batch_np * 2.0 / 255.0 - 1.0  # normalize to [-1,1]
            yield torch.tensor(batch_np)
            batch.clear()

        if terminated or truncated:
            env.reset()

# ---------------- Main Training ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps", default=False, action="store_true", help="Enable Apple MPS GPU")
    args = parser.parse_args()
    device = torch.device("mps" if args.mps else "cpu")

    # Create Atari environments
    envs = [InputWrapper(gymnasium.make(name)) for name in ("Breakout-v4", "AirRaid-v4", "Pong-v4")]
    input_shape = envs[0].observation_space.shape

    # Create networks
    net_discr = Discriminator(input_shape).to(device)
    net_gener = Generator(input_shape).to(device)

    # Optimizers & loss
    objective = nn.BCELoss()
    gen_optimizer = optim.Adam(net_gener.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(net_discr.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    

    true_labels_v = torch.ones(BATCH_SIZE, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, device=device)

    def process_batch(trainer,batch):
        gen_input_v=torch.FloatTensor(BATCH_SIZE,LATENT_VECTOR_SIZE,1,1)
        gen_input_v.normal_(0,1)
        gen_input_v=gen_input_v.to(device)
        batch_v =batch.to(device)
        gen_output_v = net_gener(gen_input_v)
        
        dis_optimizer.zero_grad()
        dis_output_true_v=net_discr(batch_v)
        dis_output_fake_v=net_discr(gen_output_v.detach())
        dis_loss = objective(dis_output_true_v,true_labels_v)+\
            objective(dis_output_fake_v,fake_labels_v)
        dis_loss.backward()
        dis_optimizer.step()
        
        gen_optimizer.zero_grad()
        dis_output_v= net_discr(gen_output_v)
        gen_loss = objective(dis_output_v,true_labels_v)
        gen_loss.backward()
        gen_optimizer.step()
        if trainer.state.iteration%SAVE_IMAGE_EVERY_ITER==0:
            fake_img = vutils.make_grid(gen_output_v.data[:64],normalize=True)
            trainer.tb.writer.add_image("fake",fake_img,trainer.state.iteration)
            real_img = vutils.make_grid(batch_v.data[:64],normalize=True)
            trainer.tb.writer.add_image("real",real_img,trainer.state.iteration)
            trainer.tb.writer.flush()
        return dis_loss.item(),gen_loss.item()
    engine = Engine(process_batch)
    tb = tb_logger.TensorboardLogger(log_dir =None)
    engine.tb = tb
    RunningAverage(output_transform=lambda out:out[1]).\
        attach(engine,"avg_loss_gen")
    RunningAverage(output_transform=lambda out:out[0]).\
        attach(engine,"avg_loss_dis")
    handler = tb_logger.OutputHandler(tag="train",metric_names=["avg_loss_gen","avg_loss_dis"])
    tb.attach(engine,log_handler=handler,event_name=Events.ITERATION_COMPLETED)
    
    @engine.on(Events.ITERATION_COMPLETED)
    def log_losses(trainer):
        if trainer.state.iteration % REPORT_EVERY_ITER==0:
            log.info("%d: gen_loss=%f ,dis_loss=%f",trainer.state.iteration,trainer.state.metrics['avg_loss_gen'],trainer.state.metrics['avg_loss_dis'])
    engine.run(data=iterate_batches(envs))