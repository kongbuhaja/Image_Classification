import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description='IC')
    parser.add_argument('--process', dest='process', type=str, default='train')
    parser.add_argument('--model', dest='model', type=str, default='resnet18')
    parser.add_argument('--path', dest='path', type=str, default='')
    parser.add_argument('--data', dest='data', type=str, default='tiny-imagenet')
    parser.add_argument('--imgsz', dest='imgsz', type=int, default=64)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=512)
    parser.add_argument('--patience', dest='patience', type=int, default=30)
    parser.add_argument('--gpus', dest='gpus', type=str, default='0', help='which device do you want to use')
    parser.add_argument('--cpus', dest='cpus', type=str, default='0-15', help='how many cores do you want to use')
    parser.add_argument('--epochs', dest='epochs', type=int, default=150)
    parser.add_argument('--wepochs', dest='wepochs', type=int, default=5)
    parser.add_argument('--eval_term', dest='eval_term', type=int, default=1)
    parser.add_argument('--init_lr', dest='init_lr', type=float, default=0.01)
    parser.add_argument('--seed', dest='seed', type=int, default=42)
    
    return parser.parse_args()