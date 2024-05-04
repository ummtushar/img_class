# CNN Image Classifier for Fresh and Spoiled Produce

This project uses a convolutional neural network (CNN) implemented in PyTorch to classify images of produce as either fresh or spoiled. The model processes grayscale images of size 1280x720 pixels.

## Project Structure

- `loading.py`: Script for loading and preprocessing the dataset.
- `main.py`: Contains the CNN model, training, and evaluation logic.
- `requirements.txt`: List of dependencies for the project.

## Setup and Installation

1. **Clone the Repository:**


2. **Create a Virtual Environment (Optional but recommended):**


3. **Install Dependencies:**



4. **Prepare the Data:**
Ensure that the `Fresh` and `Spoiled` folders are in the same directory as `loading.py`, each containing `.jpg` images of the respective class.

## Usage

To run the project, execute the following command:


This will start the training process and, upon completion, evaluate the model on the test set and print the classification report.

## Model Architecture

The CNN model is structured as follows:

- **Input Layer**: Accepts grayscale images of shape (1, 720, 1280).
- **First Convolutional Layer**: 20 filters, 5x5 kernel size.
- **First Max Pooling Layer**: 2x2 kernel size, stride 2.
- **Second Convolutional Layer**: 50 filters, 5x5 kernel size.
- **Second Max Pooling Layer**: 2x2 kernel size, stride 2.
- **Flattening Layer**: Flattens the output to feed into the fully connected layer.
- **First Fully Connected Layer**: 500 neurons.
- **Second Fully Connected Layer (Output Layer)**: 2 neurons (one for each class), followed by a log-softmax activation.

## Contributing

Contributions to this project are welcome! Here are some ways you can contribute:

- Reporting bugs
- Suggesting enhancements
- Writing or improving documentation
- Submitting pull requests to resolve issues or add features

For more details, see the CONTRIBUTING.md file (if available).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the PyTorch team for an excellent deep learning framework.
- This project was inspired by challenges faced in the food industry regarding quality control.

