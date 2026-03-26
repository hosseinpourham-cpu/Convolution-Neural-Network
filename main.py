from train.train_module import  run_training
from evaluation.evaluate import run_evaluation

if __name__ == "__main__":
    #train 
    model, test_data = run_training()

    #evaluate
    x_test,y_test = test_data
    run_evaluation(model, x_test, y_test)
