from src.read_date import train,test,sample_submission
from src.functions import lazy_EDA,data_transform,run_model,determine_threshold,analyze_results,get_submission

def main():
    lazy_EDA(train)
    train = data_transform(train)
    test = data_transform(test)
    train_pred,test_pred = run_model(train.drop("ARTIS_DURUMU",axis =1),train["ARTIS_DURUMU"],x_val = test)
    analyze_results(y_train, train_pred) # You need to check validation result but this code is directly trying to get the final result for final submission.
    # You can try all the functional steps in functions.py file.
    get_submission(sample_submission, test_pred, threshold = 0.44)


if __name__ == '__main__':
    main()