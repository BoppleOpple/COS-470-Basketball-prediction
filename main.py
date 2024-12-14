import dataset
import model
import sys
def main():
    if len(sys.argv) < 2:
        print("please provide the following arguments:")
        print("python main.py <input_path>")
        sys.exit(1)

    train_data, test_data = dataset.process(sys.argv[1])

    print(len(train_data))
    print(len(test_data))

    network = model.BasketBallModel()
    
    model.train(network, train_data, test_data)

    model.test(network, test_data)

if __name__ == "__main__":
    main()
