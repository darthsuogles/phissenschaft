#include <iostream>
#include <vector>
#include <algorithm>
#include <gflags/gflags.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <boost/filesystem.hpp>

DEFINE_bool(big_menu, true, "Include 'advanced' options in the menu listing");
DEFINE_string(languages, "english,french,german",
			  "comma-separated list of languages to offer in the 'lang' menu");

using CloudXYZI = pcl::PointCloud<pcl::PointXYZI>;

void readKittiVelodyne(const boost::filesystem::path& fileName, CloudXYZI& cloud){
    std::ifstream input(fileName.c_str(), std::ios_base::binary);
    if(!input.good()){
        std::cerr<<"Cannot open file : "<<fileName<<std::endl;
        return;
    }

    cloud.clear();
    cloud.height = 1;

    for (int i=0; input.good() && !input.eof(); i++) {
        pcl::PointXYZI point;
        input.read((char *) &point.x, 3*sizeof(float));
        input.read((char *) &point.intensity, sizeof(float));
        cloud.push_back(point);
    }
    std::cerr<<fileName.filename()<<":"<<cloud.width<<" points"<<std::endl;
    input.close();
}

int main() {

}
