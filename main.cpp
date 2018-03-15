// Sketch of one way to do a scaling study
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect.hpp>


#include "tbb/task_scheduler_init.h"
#include "tbb/tick_count.h"
#include "tbb/pipeline.h"
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>

#include <unistd.h>
#include <sys/syscall.h>
#include <stdlib.h>

#ifdef HAVE_CUDA

#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"

#endif

#include <libxml/parser.h>
#include <libxml/tree.h>

#include "H5Cpp.h"
using namespace H5;


#if defined(__APPLE__) && defined(__MACH__)

#include <cpuid.h>

#define CPUID(INFO, LEAF, SUBLEAF) __cpuid_count(LEAF, SUBLEAF, INFO[0], INFO[1], INFO[2], INFO[3])

#define GETCPU(CPU) {                              \
        uint32_t CPUInfo[4];                           \
        CPUID(CPUInfo, 1, 0);                          \
        /* CPUInfo[1] is EBX, bits 24-31 are APIC ID */ \
        if ( (CPUInfo[3] & (1 << 9)) == 0) {           \
          CPU = -1;  /* no APIC on chip */             \
        }                                              \
        else {                                         \
          CPU = (unsigned)CPUInfo[1] >> 24;                    \
        }                                              \
        if (CPU < 0) CPU = 0;                          \
      }
#else

#define GETCPU(CPU) {                                   \
        int _cpu, _status;                              \
        _status = syscall(SYS_getcpu, &_cpu, NULL, NULL); \
        if (_status != 0)                                \
            CPU = 0;                                    \
        else                                            \
            CPU = _cpu;                                  \
      }

#endif

using namespace tbb;
using namespace cv;
using namespace std;

typedef struct
{
    int classIdx;
    float *feature;
} hogFeature;

#define HOG_FEATURE_SIZE    ((sizeof(float) * 100) + sizeof(int))


int getdir (string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) 
    {
        if (dirp->d_type & DT_REG)
        {
            string name = dir+"/"+dirp->d_name;
            files.push_back(string(name));
        }
    }
    closedir(dp);
    return 0;
}


void findHogParallel(vector<string> &files) {
    int idx=0;
    int writeIdx = 0;
    RNG rng(12345);
    
    //namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    parallel_pipeline( /*max_number_of_live_token=*/16,       
        make_filter<void, Mat>(
            filter::serial,
            [&](flow_control& fc)-> Mat
            {
            	int cpu;
            	Mat image;
            	
    			GETCPU(cpu);
				if (files[idx] != *files.end())
				{
                    //cout << "CPU " << cpu  << " filter 0 checking " << files[idx] << endl;				
				    image = imread(files[idx], CV_LOAD_IMAGE_COLOR); 
                    //cout << "CPU " << cpu << " filter 0 Done " << endl;
                    idx++;
				    return image;
				}
				else
				{
				    fc.stop();
                    return image;
				}
            }    
        ) &
        make_filter<Mat,Mat>(
            filter::parallel,
            [&](Mat image)
            {
				int cpu;
				Mat grayImg, binaryImg;
				
				GETCPU(cpu);

                //cout << "CPU " << cpu << " filter 1 gray " << endl;

                cvtColor(image, grayImg, CV_RGB2GRAY);
                threshold(grayImg, binaryImg, 128, 255, THRESH_BINARY_INV);

                //cout << "CPU " << cpu << " filter 1 Done " << endl;
            	return binaryImg;
			} 
        ) &
        make_filter<Mat,Mat>(
            filter::parallel,
            [&](Mat image)
            {
				int cpu;
				vector<vector<Point> > contours;
				vector<Vec4i> hierarchy;
				Mat drawing = Mat::zeros( image.size(), CV_8UC3 );
				
				GETCPU(cpu);
                findContours( image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

                for( int i = 0; i< contours.size(); i++ )
                {
                    Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255) );
                    drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
                }

                
                //cout << "CPU " << cpu << " filter 2 Done " << endl;
            	return drawing;
			} 
        ) &  
        make_filter<Mat,void>(
            filter::parallel,
            [&](Mat image) 
            {
				int cpu;

                GETCPU(cpu);
				string name = string("./output/cup")+ to_string(idx) + string("_parallel.png");
				idx++;
				//cout << "CPU " << cpu << " filter 3 write " << name << endl;
				imwrite(name , image);
                //cout << "CPU " << cpu << " filter 3 Done " << endl;
			}
        )
    );
}

static void
print_element_names(xmlNode * a_node)
{
    xmlNode *cur_node = NULL;

    for (cur_node = a_node; cur_node; cur_node = cur_node->next) {
        if (cur_node->type == XML_ELEMENT_NODE) {
            printf("node type: Element, name: %s\n", cur_node->name);
        }

        print_element_names(cur_node->children);
    }
}

void readXML(char *fname)
{
    xmlDoc *doc = NULL;
    xmlNode *root_element = NULL;

    doc = xmlReadFile(fname, NULL, 0);

    if (doc == NULL) 
    {
        cout << "error: could not parse file " << fname << endl;
    }

    root_element = xmlDocGetRootElement(doc);
    print_element_names(root_element);
    xmlFreeDoc(doc);
    xmlCleanupParser();
}


void findHogSerial(vector<string> &files, Size roiSize) 
{
    #define FILE_NAME       "feature.h5"
    #define DATASET_NAME    "dataset"

    int idx=0;
    Mat image, grayImg, binaryImg;
    RNG rng(12345);

    hid_t hog_tid;
    hid_t feature_tid;
    hid_t dataset, space, feature_file; /* Handles */
    hsize_t numOfFeature = 0;//{files.size()};
    hsize_t max_dim = H5S_UNLIMITED;
    herr_t status;
    hsize_t featureLen;
    

    HOGDescriptor hog;
    hog.winSize = roiSize;
    hog.blockSize = Size(16, 16);
    hog.blockStride = Size(8, 8);
    hog.cellSize = Size(4, 4);

    // create a new HDF5 file
    feature_file = H5Fcreate(FILE_NAME, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    // create HDF5 space
    space = H5Screate_simple(1, &numOfFeature, &max_dim);
    // initialize array dimension with feature descriptor len and create array data type
    featureLen = (hsize_t)hog.getDescriptorSize();
    cout << "getDescriptorSize return " << featureLen << endl;
    // create a tid of feature vector
    feature_tid = H5Tarray_create(H5T_NATIVE_FLOAT, 1, &featureLen);

    // create a compound data type with a class ID and a feature descriptor
    hog_tid = H5Tcreate (H5T_COMPOUND, sizeof(int) + sizeof(float) * featureLen);
    H5Tinsert(hog_tid, "classIdx", HOFFSET(hogFeature, classIdx), H5T_NATIVE_INT);
    H5Tinsert(hog_tid, "feature", 4, feature_tid);    

    // init a prop list for chunked
    hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_layout(plist, H5D_CHUNKED);
    hsize_t chunk_size = 5;
    H5Pset_chunk(plist, 1, &chunk_size);    
    // create dataset with chunked prop list
    dataset = H5Dcreate(feature_file, DATASET_NAME, hog_tid, space, H5P_DEFAULT, plist, H5P_DEFAULT);    
    // close plist
    H5Pclose(plist);
    // space is obsolete, so close it
    H5Sclose(space);
    
    // create a continuous buffer for HDF5 write
    char *hogFeatureBuff = (char *)malloc(sizeof(int) + sizeof(float) * featureLen);
    float *pFeature = (float *) &hogFeatureBuff[sizeof(int)];
    int *classIdx = (int *) hogFeatureBuff;

    vector< float > descriptors;

    tbb::tick_count::interval_t t0_threshold((double)0), t0_gray((double)0), t0_findContour((double)0);;
    
    hsize_t index = 0, memDims = 1;
    hid_t mem_space = H5Screate_simple(1, &memDims, NULL);
    for (vector<string>::iterator it = files.begin() ; it != files.end(); ++it)
    {
        tbb::tick_count t0;
        Rect r(0, 0, roiSize.width, roiSize.height);
        
        image = imread(*it, CV_LOAD_IMAGE_COLOR); 

        *classIdx = index;
        t0 = tbb::tick_count::now();
        cvtColor(image(r), grayImg, CV_RGB2GRAY);
        t0_gray += (tbb::tick_count::now()-t0);
        hog.winSize = Size(grayImg.cols, grayImg.rows);
        hog.compute( grayImg, descriptors, Size( 0, 0 ), Size( 0, 0 ) );
        std::copy(descriptors.begin(), descriptors.end(), pFeature);

        cout << "descriptor size is " << descriptors.size() << endl;

        // increase the extent of dataset by 1
        numOfFeature++;
        H5Dset_extent(dataset, &numOfFeature);
        space = H5Dget_space(dataset);

        H5Sselect_elements(space, H5S_SELECT_SET, 1, &index);
        status = H5Dwrite(dataset, hog_tid, mem_space, space, H5P_DEFAULT, hogFeatureBuff);
        index ++;
    }

    H5Tclose(hog_tid);
    H5Tclose(feature_tid);
    H5Sclose(space);
    H5Dclose(dataset);
    H5Fclose(feature_file);

}


void saveFeatureToHDF5_Test()
{
    #define IMG_NUMBER  10

    #define FILE_NAME       "feature.h5"
    #define DATASET_NAME    "dataset"
    hid_t hog_tid;
    hid_t array_tid;
    hid_t dataset, space, file; /* Handles */
    hsize_t featureLen = 100;
    hsize_t numOfFeature = 0;
    hsize_t max_dim = H5S_UNLIMITED;
    herr_t status;

    file = H5Fcreate(FILE_NAME, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    space = H5Screate_simple(1, &numOfFeature, &max_dim);
    array_tid = H5Tarray_create(H5T_NATIVE_FLOAT, 1, &featureLen);

    hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_layout(plist, H5D_CHUNKED);
    hsize_t chunk_size = 5;
    H5Pset_chunk(plist, 1, &chunk_size);



    hog_tid = H5Tcreate (H5T_COMPOUND, sizeof(int) + sizeof(float) * 100);

    H5Tinsert(hog_tid, "classIdx", HOFFSET(hogFeature, classIdx), H5T_NATIVE_INT);
    H5Tinsert(hog_tid, "feature", 4, array_tid);

    cout << "HOFFSET classIdx " << HOFFSET(hogFeature, classIdx) << endl;
    cout << "HOFFSET feature " << HOFFSET(hogFeature, feature) << endl;

    dataset = H5Dcreate(file, DATASET_NAME, hog_tid, space, H5P_DEFAULT, plist, H5P_DEFAULT);

    H5Pclose(plist);


    hsize_t dims_out;
    H5Sget_simple_extent_dims(space, &dims_out, NULL);
    hsize_t npoints = H5Sget_simple_extent_npoints(space);
    cout << __FUNCTION__ << endl;
    cout << "   dims_out=" << dims_out << ",  npoints = " << npoints << endl;

    H5Sclose(space);

    //status = H5Dwrite(dataset, hog_tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, hogFeatureBuff);

    char *hogFeatureBuff = (char *)malloc(HOG_FEATURE_SIZE);
    hogFeature *featureArray;

    hsize_t mDims=1;
    hid_t mem_space = H5Screate_simple(1, &mDims, NULL);

    int num = 0;
    hsize_t new_dims=1;
    cout << "hogFeatureBuff = " << hogFeatureBuff << endl;

    int *classIdx = (int *)hogFeatureBuff;
    float *feature = (float *)&hogFeatureBuff[sizeof(int)];
    for (hsize_t i = 0; i < IMG_NUMBER; i++)
    {
        *classIdx = i;
        num = i * 1000;

        for (int j = 0; j < 100; j++)
        {
            feature[j] = num * 0.01;
            num += 1;
        }
        
        H5Dset_extent(dataset, &new_dims);
        space = H5Dget_space(dataset);

        H5Sselect_elements(space, H5S_SELECT_SET, 1, &i);

        H5Sget_simple_extent_dims(space, &dims_out, NULL);
        npoints = H5Sget_simple_extent_npoints(space);

        H5Dwrite(dataset, hog_tid, mem_space, space, H5P_DEFAULT, hogFeatureBuff);

        cout << "Extend Dataset" << endl;
        cout << "   dims_out=" << dims_out << ",  npoints = " << npoints << endl;
        new_dims++;
        //break;
    }


    H5Tclose(hog_tid);
    H5Tclose(array_tid);
    H5Sclose(space);
    H5Dclose(dataset);
    H5Fclose(file);
    
}


void loadFeatureToHDF5()
{
    #define IMG_NUMBER  10

    #define FILE_NAME       "feature.h5"
    #define DATASET_NAME    "dataset"
    hid_t hog_tid;
    hid_t array_tid;
    hid_t dataset, space, file; /* Handles */
    hid_t memSpace;
    hsize_t dims_out[] = {0, 0};
    hsize_t out_dim[] = {1};
    hsize_t coords[] = {0};
    herr_t status;
    hsize_t size, dataSize;
    hsize_t rank, npoints;


    //H5File file(FILE_NAME, H5F_ACC_TRUNC);

    file = H5Fopen(FILE_NAME, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset = H5Dopen(file, DATASET_NAME, H5P_DEFAULT);
    hog_tid = H5Dget_type(dataset); 

    space = H5Dget_space(dataset);    /* dataspace handle */
    rank = H5Sget_simple_extent_ndims(space);
    size = H5Dget_storage_size(dataset);
    H5Sget_simple_extent_dims(space, dims_out, NULL);   
    npoints = H5Sget_simple_extent_npoints(space);
    dataSize = H5Tget_size(hog_tid);

    cout << __FUNCTION__ << " size is " << size << endl;
    cout << __FUNCTION__ << " rank is " << rank << endl;    
    cout << __FUNCTION__ << " dims_out is " << dims_out[0] << "," << dims_out[1] << endl;    
    cout << __FUNCTION__ << " npoints is " << npoints << endl;
    cout << __FUNCTION__ << " dataSize is " << dataSize << endl;    

    memSpace = H5Screate_simple(1, out_dim, NULL);

    coords[0] = 0;
    status = H5Sselect_elements(space, H5S_SELECT_SET, 1, (const hsize_t *)&coords);
    hogFeature *f = (hogFeature *)malloc(dataSize);

    status = H5Dread(dataset, hog_tid, memSpace, space, H5P_DEFAULT, f);

    cout << "class: " << f->classIdx << endl;
    float *feature = (float *) ((char *)f + sizeof(int));
    for (int i = 0; i < 10; i++)
    {
        cout << feature[i] << ' ';
    }
    cout << endl;

    H5Tclose(hog_tid);
    H5Sclose(space);
    H5Sclose(memSpace);
    H5Dclose(dataset);
    H5Fclose(file);

}

int main() 
{
	vector<string> imgNameList;
	getdir("./samples", imgNameList);
	cout << "img count = " << imgNameList.size() << endl;
    tbb::tick_count t0, t1;

    readXML("./samples/img128_0.xml");

    t0 = tbb::tick_count::now();
    findHogSerial(imgNameList, Size(320, 240)); 
    t1 = tbb::tick_count::now();
    cout << "findHogSerial takes " << (t1 - t0).seconds() << endl;

    //saveFeatureToHDF5();
    loadFeatureToHDF5();

    return 0;
}


