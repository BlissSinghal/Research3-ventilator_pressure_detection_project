import matplotlib.pyplot as plt

def make_plot(x_test, y_test, y_preds):

    x_test = x_test.tolist()
    y_test = y_test.tolist()
    y_preds = y_preds.tolist()
    print(len(x_test) == len(y_preds))
    
    fig = plt.figure()
    plt.scatter(x_test, y_test)
    plt.plot(x_test, y_preds, color = "red")
    plt.savefig(f"graphs/results.png")
