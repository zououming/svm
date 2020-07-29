#include <dirent.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace ml;

void load_data(std::string path, Mat& set, Mat& labels);
void SVM_train();
void prediction(std::string img_path, std::string model_path);

#define training_mode 1
#define prediction_mode 2
Mat train_set, validation_set;
Mat train_labels, validation_labels;
int model;

int main(int argc, char* argv[]) {
    std::string train_path = "../train";
    std::string validation_path = "../validation";
    std::string prediction_path;
    std::string model_path = "../model/svm_arms.xml";

    if( argc > 1 ){
        for( int i = 1; i < argc; i++ ){
            if( strcmp(argv[i], "train") == 0 ) {
                train_path = argv[++i];
                model = training_mode;
            }
            else if( strcmp(argv[i], "validation") == 0 ) {
                validation_path = argv[++i];
                model = training_mode;
            }
            else if( strcmp(argv[i], "prediction") == 0 ) {
                prediction_path = argv[++i];
                model = prediction_mode;
            }
            else if( strcmp(argv[i], "model") == 0 ) {
                model_path = argv[++i];
                model = prediction_mode;
            }
        }

    }

    if( model == training_mode ) {
        load_data(train_path, train_set, train_labels);
        load_data(validation_path, validation_set, validation_labels);
        SVM_train();
    }
    else if( model == prediction_mode ) {
        prediction(prediction_path, model_path);
    }
    return 0;
}

void load_data(std::string path, Mat& set, Mat& labels){
    DIR* dir = opendir(path.c_str());
    dirent* class_name = NULL;
    while( (class_name = readdir(dir)) != NULL ) {
        if (class_name->d_name[0] != '.') {
            std::string class_path = std::string(path) + '/' + class_name->d_name;
            DIR *class_dir = opendir(class_path.data());
            dirent *file_name = NULL;
            while ((file_name = readdir(class_dir)) != NULL)
                if (file_name->d_name[0] != '.') {
                    std::string file_path = class_path + '/' + std::string(file_name->d_name);
                    Mat img = imread(file_path);
		    resize(img, img, Size(25, 25));
                    cvtColor(img, img, COLOR_BGR2GRAY);
		    threshold(img, img, 100, 255, THRESH_OTSU);
                    Mat vector = img.reshape(1,1);
                    set.push_back(vector);
                    labels.push_back(atoi(class_name->d_name));
                }
        }
    }
    set.convertTo(set, CV_32FC2);
    labels.convertTo(labels, CV_32SC1);
}

void SVM_train(){
    Ptr<SVM> svm = SVM::create();//以下是设置SVM训练模型的配置
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setGamma(1);
    svm->setC(1);
    svm->setCoef0(0);
    svm->setNu(0);
    svm->setP(0);
    svm->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 20000, 0.0001));

    Ptr<TrainData>svm_set = TrainData::create(train_set, ROW_SAMPLE, train_labels);
    svm->train(svm_set);
    printf("Training completed!\n");
    float count = 0;
    for( int i = 0; i < validation_labels.size[0]; i++){
        int predict_result = svm->predict(validation_set.row(i));
        if( predict_result == validation_labels.at<Vec3b>(i, 0)[0] )
            count++;
    }

    float accuracy = count / validation_labels.size[0];
    printf("Accuracy: %f\n",accuracy);
    if( accuracy > 0.8 ){
        printf("Save the model? Y/n : ");
        char save;
        std::cin >> save;
        std::string model_path = "../model/", model_name;
        if ( save == 'Y' || save == 'y' ){
            printf("Enter the model name (*.xml): ");
            std::cin >> model_name;
            svm->save(model_path + model_name);
            std::cout << "Saved to " << model_path + model_name << std::endl;
        }
    }
}

void prediction(std::string img_path, std::string model_path){
    Ptr<SVM> svm = SVM::load(model_path);
    Mat img = imread(img_path);
    resize(img, img, Size(25, 25));
    cvtColor(img, img, COLOR_BGR2GRAY);
    threshold(img, img, 100, 255, THRESH_OTSU);
    Mat vector = img.reshape(1,1);
    vector.convertTo(vector, CV_32FC2);

    std::cout<<"Prediction: "<<svm->predict(vector)<<std::endl;
}
