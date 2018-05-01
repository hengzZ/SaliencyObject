/*list files in target directory on Linux*/
#include <sys/types.h>
#include <dirent.h>
#include <cstdio>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>


void GetAllFiles(const std::string target_path, std::vector<std::string>& files)
{
    files.clear();
    struct dirent *ptr;
    DIR *dir;
    dir = opendir(target_path.c_str());
    if(NULL == dir) return;
    while((ptr = readdir(dir)) != NULL){
        if(ptr->d_name[0] == '.')
            continue;
        files.push_back(ptr->d_name);
    }
}


void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;

    while(std::string::npos != pos2){
        v.push_back(s.substr(pos1, pos2-pos1));

        pos1 = pos2+c.size();
        pos2 = s.find(c, pos1);
    }
    if(pos1 != s.length())
        v.push_back(s.substr(pos1));
}

