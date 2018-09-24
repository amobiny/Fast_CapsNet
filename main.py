import tensorflow as tf
from config import args
import os
from utils import write_spec

if args.model == 'original_capsule':
    from Original_CapsNet import OrigCapsNet as Model
elif args.model == 'vector_capsule':
    from FastCapsNet import FastCapsNet3D as Model


def main(_):
    if args.mode not in ['train', 'test']:
        print('invalid mode: ', args.mode)
        print("Please input a mode: train or test")
    elif args.mode == 'train' or args.mode == 'test':
        model = Model(tf.Session(), args)
        if not os.path.exists(args.modeldir+args.run_name):
            os.makedirs(args.modeldir+args.run_name)
        if not os.path.exists(args.logdir+args.run_name):
            os.makedirs(args.logdir+args.run_name)
        if args.mode == 'train':
            write_spec(args)
            model.train()
        elif args.mode == 'test':
            model.test(args.reload_step)


if __name__ == '__main__':
    # configure which gpu or cpu to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.app.run()
