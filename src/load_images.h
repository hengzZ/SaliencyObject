#ifndef LOAD_IMAGES_H
#define LOAD_IMAGES_H
#include <string>
#include <vector>


void GetAllFiles(const std::string target_path, std::vector<std::string>& files);

void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c);

#endif // LOAD_IMAGES_H
