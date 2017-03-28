#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include "matrix.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

template<typename Type>
void print_matrix(Matrix<Type> matrix);

template<typename Type>
void print_vector(vector<Type> vector);

const uint height_parts = 4;
const uint width_parts = 8;
const uint hist_parts = 24;
const uint pyramid_par = 2;
const int DEG_RANGE = 360;

// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());
    
    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);
    
    while(stream >> filename >> label) {
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }
    
    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
        // Create image
        BMP* image = new BMP();
        // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
        // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second)); //image, label
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels,
                     const string& prediction_file) {
    // Check that list of files and list of labels has equal size
    assert(file_list.size() == labels.size());
    // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());
    
    // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}

void make_grayscale_matrix(Matrix<int> &matrix_image_grayscale, BMP* bmp_image) {
    for (uint i = 0; i < matrix_image_grayscale.n_rows; i++) {
        for (uint j = 0; j < matrix_image_grayscale.n_cols; j++) {
            RGBApixel pixel = bmp_image -> GetPixel(j, i);
            int s = static_cast<int>(0.299 * pixel.Red + 0.114 * pixel.Blue + 0.587 * pixel.Green);
            matrix_image_grayscale(i, j) = s;
        }
    }
}

void sobel_filter_hor(const Matrix<int> &matrix_image_grayscale, Matrix<int> &matrix_sobel_hor) {
    
    for (uint i = 0; i < matrix_image_grayscale.n_rows; i++) {
        for (uint j = 0; j < matrix_image_grayscale.n_cols; j++) {
            int x = matrix_image_grayscale(i, j);
            int y;
            if (j >= matrix_image_grayscale.n_cols - 2) {
                y = matrix_image_grayscale(i, matrix_image_grayscale.n_cols - 1);
            } else {
                y = matrix_image_grayscale(i, j + 2);
            }
            matrix_sobel_hor(i, j) = y - x;
        }
    }
}

void sobel_filter_ver(const Matrix<int> &matrix_image_grayscale, Matrix<int> &matrix_sobel_ver) {
    
    for (uint i = 0; i < matrix_image_grayscale.n_rows; i++) {
        for (uint j = 0; j < matrix_image_grayscale.n_cols; j++) {
            int x = matrix_image_grayscale(i, j);
            int y;
            if (i >= matrix_image_grayscale.n_rows - 2) {
                y = matrix_image_grayscale(matrix_image_grayscale.n_rows - 1, j);
            } else {
                y = matrix_image_grayscale(i + 2, j);
            }
            matrix_sobel_ver(i, j) = x - y;
        }
    }
}

void resize(BMP* image, BMP* resized_image) {
    
    uint height = image->TellHeight();
    uint width = image->TellWidth();
    
    uint newHeight = ((height + (height_parts - 1)) / height_parts) * height_parts;
    uint newWidth = ((width + (width_parts - 1)) / width_parts) * width_parts;
    
    resized_image->SetSize(newWidth, newHeight);
    
    float k1 = static_cast<float>(height)/ newHeight;
    float k2 = static_cast<float>(width) / newWidth;
    
    for (uint i = 0; i < newWidth; ++i) {
        for (uint j = 0; j < newHeight; ++j) {
            resized_image->SetPixel(i, j, image->GetPixel(round(i * k2), round(j * k1)));
        }
    }
}

void make_module_and_direction_matrix(const Matrix<int> &matrix_sobel_hor, const Matrix<int> &matrix_sobel_ver,
                                      Matrix<float> &module_grad_matrix, Matrix<int> &direction_grad_matrix) {
    for (uint i = 0; i < module_grad_matrix.n_rows; ++i) {
        for (uint j = 0; j < module_grad_matrix.n_cols; ++j) {
            
            module_grad_matrix(i, j) =
            static_cast<float>(sqrt(matrix_sobel_hor(i, j) * matrix_sobel_hor(i, j) + matrix_sobel_ver(i, j) * matrix_sobel_ver(i, j)));
            
            int angle = (matrix_sobel_hor(i, j) * matrix_sobel_ver(i, j) != 0) ?
            static_cast<int>(atan(static_cast<float>(abs(matrix_sobel_ver(i, j))) / abs(matrix_sobel_hor(i, j))) / M_PI * (DEG_RANGE / 2)) : 0;
            
            if (matrix_sobel_hor(i, j) > 0 && matrix_sobel_ver(i, j) > 0) {
                direction_grad_matrix(i, j) = angle;
            } else if (matrix_sobel_hor(i, j) < 0 && matrix_sobel_ver(i, j) > 0) {
                direction_grad_matrix(i, j) = DEG_RANGE / 2 - angle;
            } else if (matrix_sobel_hor(i, j) < 0 && matrix_sobel_ver(i, j) < 0) {
                direction_grad_matrix(i, j) = DEG_RANGE / 2 + angle;
            } else if (matrix_sobel_hor(i, j) > 0 && matrix_sobel_ver(i, j) < 0) {
                direction_grad_matrix(i, j) = DEG_RANGE - angle;
            } else {
                direction_grad_matrix(i, j) = 0;
            }
        }
    }
}
//получаем гистограмму ориентированных градиентов пикселей выбранной клетки
void make_cell_hist(const Matrix<float> &cell_mod, const Matrix<int> &cell_dir, vector<float> &cell_hist) {
    
    for (uint i = 0; i < cell_mod.n_rows; ++i) {
        for (uint j = 0; j < cell_mod.n_cols; j++) {
            uint segment = static_cast<uint>(trunc(static_cast<float>(cell_dir(i, j)) / DEG_RANGE * cell_hist.size()));
            cell_hist[segment] += cell_mod(i, j);
        }
    }
}

void vector_normalization(vector<float> &vector) {
    double squares_sum = 0;
    for (uint i = 0; i < vector.size(); i++) {
        squares_sum += vector[i] * vector[i];
    }
    double norm = sqrt(squares_sum);
    if (static_cast<int>(norm) != 0) {
        for (uint i = 0; i < vector.size(); i++) {
            vector[i] /= norm;
        }
    }
}

void make_descriptor(const Matrix<float> &module_grad_matrix, const Matrix<int> &direction_grad_matrix, vector<float> &descriptor) {
    
    uint cell_height = direction_grad_matrix.n_rows / height_parts;
    uint cell_width = direction_grad_matrix.n_cols / width_parts;
    
    //разделяем матрицу на ячейки (матрица - height_parts * width_parts)
    for (uint i = 0; i < height_parts; ++i) {
        for (uint j = 0; j < width_parts; ++j) {
            
            Matrix<int> cell_dir = direction_grad_matrix.submatrix(i * cell_height, j * cell_width, cell_height, cell_width);
            Matrix<float> cell_mod = module_grad_matrix.submatrix(i * cell_height, j * cell_width, cell_height, cell_width);
            vector<float> cell_hist(hist_parts);
            
            make_cell_hist(cell_mod, cell_dir, cell_hist);
            vector_normalization(cell_hist);
            descriptor.insert(descriptor.end(), cell_hist.begin(), cell_hist.end());
        }
    }
}

void hog(BMP* image, vector<float> &descriptor) {
    
    uint height = image->TellHeight();
    uint width = image->TellWidth();
    
    Matrix<int> matrix_image_grayscale(height, width);
    make_grayscale_matrix(matrix_image_grayscale, image);
    
    Matrix<int> matrix_sobel_hor(height, width);
    sobel_filter_hor(matrix_image_grayscale, matrix_sobel_hor);
    
    Matrix<int> matrix_sobel_ver(height, width);
    sobel_filter_ver(matrix_image_grayscale, matrix_sobel_ver);
    
    Matrix<float> module_grad_matrix(height, width);
    Matrix<int> direction_grad_matrix(height, width);
    make_module_and_direction_matrix(matrix_sobel_hor, matrix_sobel_ver, module_grad_matrix, direction_grad_matrix);
    
    make_descriptor(module_grad_matrix, direction_grad_matrix, descriptor);
}

void pyramid(BMP* image, vector<float> &descriptor) {
    
    uint cell_height = image->TellHeight() / pyramid_par;
    uint cell_width = image->TellWidth() / pyramid_par;
    
    //изображение делится на pyramid_par части и для каждой считается hog
    for (uint i = 0; i < pyramid_par; ++i) {
        for (uint j = 0; j < pyramid_par; ++j) {
            BMP tmp;
            tmp.SetSize(1,1);
            BMP* cell_bmp = &tmp;
            cell_bmp->SetSize(cell_width, cell_height);
            for (uint m = 0; m < cell_width; ++m) {
                for (uint n = 0; n < cell_height; ++n) {
                    cell_bmp->SetPixel(m, n, image->GetPixel(i * cell_width + m, j * cell_height + n));
                }
            }
            vector<float> cell_descriptor(0);
            hog(cell_bmp, cell_descriptor);
            
            descriptor.insert(descriptor.end(), cell_descriptor.begin(), cell_descriptor.end());
        }
    }
}

void make_avg_color_rgb_vector(BMP* cell_bmp, vector<float> &avg_color_rgb) {
    
    uint sum_r = 0;
    uint sum_g = 0;
    uint sum_b = 0;
    for (int i = 0; i < cell_bmp->TellWidth(); ++i) {
        for (int j = 0; j < cell_bmp->TellHeight(); ++j) {
            RGBApixel pixel = cell_bmp->GetPixel(i, j);
            sum_r += pixel.Red;
            sum_g += pixel.Green;
            sum_b += pixel.Blue;
        }
    }
    uint pixels_number = cell_bmp->TellWidth() * cell_bmp->TellHeight();
    
    avg_color_rgb.push_back(static_cast<float>(sum_r) / pixels_number / 255);
    avg_color_rgb.push_back(static_cast<float>(sum_g) / pixels_number / 255);
    avg_color_rgb.push_back(static_cast<float>(sum_b) / pixels_number / 255);
}

void color(BMP* image, vector<float> &descriptor) {
    
    uint parts_number = 8;
    uint cell_height = image->TellHeight() / parts_number;
    uint cell_width = image->TellWidth() / parts_number;
    
    for (uint i = 0; i < parts_number; ++i) {
        for (uint j = 0; j < parts_number; ++j) {
            BMP tmp;
            tmp.SetSize(1,1);
            BMP* cell_bmp = &tmp;
            cell_bmp->SetSize(cell_width, cell_height);
            
            for (uint m = 0; m < cell_width; ++m) {
                for (uint n = 0; n < cell_height; ++n) {
                    cell_bmp->SetPixel(m, n, image->GetPixel(i * cell_width + m, j * cell_height + n));
                }
            }

            vector<float> avg_color_rgb(0);
            make_avg_color_rgb_vector(cell_bmp, avg_color_rgb);
            
            //добавляем в дескриптор вектор из средних значений компонент цветов для ячейки
            descriptor.insert(descriptor.end(), avg_color_rgb.begin(), avg_color_rgb.end());
        }
    }
}

void kernel_svm(vector<float> &descriptor) {
    const int N = 1;
    const float L = 0.5;
    const float E = 1e-10;
    
    vector<float> tmp(0);
    
    for (uint i = 0; i < descriptor.size(); ++i) {
        vector<float> component(0);
        
        for (int k = -N; k <= N; ++k) {
            float lambda_value = k * L;
            float expr = M_PI * lambda_value;
            float par1 = sqrt(descriptor[i] * 2 / (exp(expr) + exp(-expr)));
            float par2 = lambda_value * ((descriptor[i] > E) ? log(descriptor[i]) : E);
            float re = par1 * cos(par2);
            float im = par1 * sin(par2);
            
            component.push_back(re);
            component.push_back(im);
        }
        tmp.insert(tmp.end(), component.begin(), component.end());
        component.clear();
    }
    descriptor.insert(descriptor.end(), tmp.begin(), tmp.end());
}

// Exatract features from dataset.
// You should implement this function by yourself =)
void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
        
        //cout << "number: "<<image_idx << endl;
        vector<float> descriptor(0);
        
        bool color_mod = true;
        bool pyramid_mod = true;
        bool kernel_mod = true;

        BMP tmp;
        tmp.SetSize(1,1);
        BMP* resized_image = &tmp;
        
        resize(data_set[image_idx].first, resized_image);
        hog(resized_image, descriptor);
        
        if (pyramid_mod) {
            pyramid(resized_image, descriptor);
        }
        if (color_mod) {
            color(resized_image, descriptor);
        }
        if (kernel_mod) {
            kernel_svm(descriptor);
        }
        features->push_back(make_pair(descriptor, data_set[image_idx].second));
    }
}

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
    // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
    // Clear dataset
    data_set->clear();
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
    // List of image file names and its labels
    TFileList file_list;
    // Structure of images and its labels
    TDataSet data_set;
    // Structure of features of images and its labels
    TFeatures features;
    // Model which would be trained
    TModel model;
    // Parameters of classifier
    TClassifierParams params;
    
    // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
    // Load images
    LoadImages(file_list, &data_set);
    // Extract features from images
    ExtractFeatures(data_set, &features);
    
    
    
    // PLACE YOUR CODE HERE
    // You can change parameters of classifier here
    params.C = 0.1;
    TClassifier classifier(params);
    
    // Train classifier
    classifier.Train(features, &model);
    // Save model to file
    model.Save(model_file);
    // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
    // List of image file names and its labels
    TFileList file_list;
    // Structure of images and its labels
    TDataSet data_set;
    // Structure of features of images and its labels
    TFeatures features;
    // List of image labels
    TLabels labels;
    
    // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
    // Load images
    LoadImages(file_list, &data_set);
    // Extract features from images
    ExtractFeatures(data_set, &features);
    
    // Classifier
    TClassifier classifier = TClassifier(TClassifierParams());
    // Trained model
    TModel model;
    // Load model from file
    model.Load(model_file);
    // Predict images by its features using 'model' and store predictions
    // to 'labels'
    classifier.Predict(features, model, &labels);
    
    // Save predictions
    SavePredictions(file_list, labels, prediction_file);
    // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    
    // Command line options parser
    ArgvParser cmd;
    // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
    // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
    // Add other options
    cmd.defineOption("data_set", "File with dataset",
                     ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
                     ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
                     ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
    
    // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");
    
    // Parse options
    int result = cmd.parse(argc, argv);
    
    // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }
    
    // Get values
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");
    
    // If we need to train classifier
    if (train)
        
        TrainClassifier(data_file, model_file);
    // If we need to predict data
    if (predict) {
        // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
        // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
        // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}
