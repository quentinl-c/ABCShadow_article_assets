//
//  GraphWrapper.cpp
//  ABC_Shadow_cpp
//
//  Created by Quentin on 02/08/2019.
//  Copyright © 2019 Quentin. All rights reserved.
//

#include "GraphWrapper.hpp"

GraphWrapper::GraphWrapper(const int inSize, const int outSize) {
    vector<vector<Node>> mat = vector<vector<Node>>(inSize);
    nodeList = vector<Node>();
    this->inSize = inSize;
    this->outSize = outSize;
    
    for (int i{0}; i < inSize; i++) {
        mat[i] = vector<Node>(inSize + outSize);

        for (int j{i + 1}; j < inSize; j++) {
            mat[i][j] = Node(i,j, NodeType::INTRA);
        }

        for(int j{inSize}; j < inSize + outSize; j ++) {
            mat[i][j] = Node(i,j, NodeType::INTER);
        }
    }
    for(int i{0}; i < inSize; i++) {
        for(int j{i+1}; j < inSize; j++) {
            //std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";
            //std::cout << i << ","<< j << " voisin de :\n";
            for(int k{0}; k < inSize; k ++) {
                if (k < i) {
                    //std::cout << " 1 "<< k << "," << i << " | ";
                    mat[i][j].addNeighbour(mat[k][i].getLabel());
                }else if (k > i && k != j) {
                    //std::cout << " 2 " << i << "," << k << " | ";
                    mat[i][j].addNeighbour(mat[i][k].getLabel());
                }
                
                if (k < j && k != i) {
                    //std::cout << " 3 " << k << "," << j << " | ";
                    mat[i][j].addNeighbour(mat[k][j].getLabel());
                } else if (k > j) {
                    //std::cout << " 4 " << j << "," << k << " | ";
                    mat[i][j].addNeighbour(mat[j][k].getLabel());
                }
            }
            
            for(int k{inSize}; k < inSize + outSize; k++) {
                mat[i][j].addNeighbour(mat[i][k].getLabel());
                mat[i][k].addNeighbour(mat[i][j].getLabel());
                
                mat[i][j].addNeighbour(mat[j][k].getLabel());
                mat[j][k].addNeighbour(mat[i][j].getLabel());
            }
            nodeList.push_back(mat[i][j]);
            //std::cout << "\n";
        }
    }
    
    for (int i{0}; i < inSize; i++) {
        for(int j{inSize}; j < inSize + outSize; j++) {
            for(int k{inSize}; k < inSize + outSize; k++) {
                if(k != j) {
                    mat[i][j].addNeighbour(mat[i][k].getLabel());
                }
            }
        }
    }
    for (int i{0}; i < inSize; i++) {
        for(int j{inSize}; j < inSize + outSize; j ++) {
           nodeList.push_back(mat[i][j]);
        }
    }
    
}

GraphWrapper::~GraphWrapper() {}

void GraphWrapper::empty() {
    for(auto &n : nodeList) {
        n.updateNodeState(State::DISABLED);
    }
}


vector<Node> &GraphWrapper::getNodes() {
    return nodeList;
}

Stats GraphWrapper::getInteractionStats() {
    Stats stats{};
    for(auto n : getNodes()) {
        stats +=  n.getInteractionStats();
    }
    return stats / 2;
}
