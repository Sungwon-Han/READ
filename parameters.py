import argparse

def data_pruning_parser():
    parser = argparse.ArgumentParser(description='Data Pruning Parser')
    
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='learning rate')
    parser.add_argument('--batch-size', default=50, type=int, help='batch size')
   
    parser.add_argument('--epochs', default=60, type=int, help='total epochs')
    parser.add_argument('--checkpoint-epochs', default=10, type=int, help='checkpoint frequency')
    parser.add_argument('--evaluation-epochs', default=10, type=int, help='evaluation frequency') 
    parser.add_argument('--workers', default=4, type=int, help='number of workers') 
    
    parser.add_argument('--load', dest='load', action='store_true', help='load trained model')
    parser.add_argument('--modelurl', type=str, default='./model/pretrained_remover.ckpt', help='model path')
    
    parser.add_argument('--train-path', type=str, help='Train images directory path to remove uninhabited areas') 
    parser.add_argument('--test-path', type=str, help='Test images directory path to remove uninhabited areas') 
    return parser.parse_args()



def extract_embeddings_parser():
    parser = argparse.ArgumentParser(description='Extract Embeddings Parser')

    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='learning rate')
    parser.add_argument('--batch-size', default=50, type=int, help='batch size')
   
    parser.add_argument('--epochs', default=10, type=int, help='total epochs')
    parser.add_argument('--checkpoint-epochs', default=10, type=int, help='checkpoint frequency')
    parser.add_argument('--evaluation-epochs', default=10, type=int, help='evaluation frequency') 
    parser.add_argument('--workers', default=4, type=int, help='number of workers') 
    
    parser.add_argument('--load', dest='load', action='store_true', help='load trained model')
    parser.add_argument('--modelurl', type=str, default='./model/pretrained_embedding.ckpt', help='model path')
    return parser.parse_args()


def predict_demographics_parser():
    parser = argparse.ArgumentParser(description='Predict Demographics Parser')
    
    parser.add_argument('--idx', default=0, type=int, help='select which demographics to predict, 0 to 51')
    return parser.parse_args()
