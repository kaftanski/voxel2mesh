import logging

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, net, trainloader, optimizer, numb_of_itrs, eval_every, save_path, evaluator):

        self.net = net
        self.trainloader = trainloader
        self.optimizer = optimizer

        self.numb_of_itrs = numb_of_itrs
        self.eval_every = eval_every
        self.save_path = save_path

        self.evaluator = evaluator

        self.writer = SummaryWriter(log_dir=save_path)

    def training_step(self, data, epoch):
        # Get the minibatch

        self.optimizer.zero_grad()
        loss, log = self.net.loss(data, epoch)
        loss.backward()
        self.optimizer.step()
        # embed()

        return log

    def train(self, start_iteration=1):

        print("Start training...") 
 
        self.net = self.net.train()
        iteration = start_iteration 

        print_every = 1
        for epoch in range(10000000):  # loop over the dataset multiple times

            log_vals = {}
            for itr, data in enumerate(self.trainloader):

                # training step 
                loss = self.training_step(data, iteration)

                for key, value in loss.items():
                    v = value / len(self.trainloader)
                    if key not in log_vals.keys():
                        log_vals[key] = v
                    else:
                        log_vals[key] += v

                iteration = iteration + 1

                if iteration % self.eval_every == self.eval_every-1:  # print every K epochs
                    self.evaluator.evaluate(iteration)

            for key, value in log_vals.items():
                self.writer.add_scalar('train/' + key, value, iteration)

            if iteration > self.numb_of_itrs:
                break


        logger.info("... end training!")
 