//
//  Node.cpp
//  ABC_Shadow_cpp
//
//  Created by Quentin on 16/10/2019.
//  Copyright © 2019 Quentin. All rights reserved.
//

#include "Node.hpp"
Node::Node() {
    nodeName = NodeName(0, 0);
    label = std::make_shared<NodeType>(NodeType::NIL);
    neighbourhood = vector<shared_ptr<NodeType>>();
}

Node::Node(int i, int j, NodeType activationVal)  {
    this->nodeName = NodeName(i, j);
    this->label = std::make_shared<NodeType>(NodeType::NIL);
    this->neighbourhood = vector<shared_ptr<NodeType>>();
    this->activationVal = activationVal;
}


void Node::addNeighbour(const shared_ptr<NodeType> neighbourType) {
    neighbourhood.push_back(neighbourType);
}

void Node::addNeighbours(vector<Node> &neighbourhoodNode){
    if(neighbourhoodNode.size() != 0) {
        
   // std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";
   // std::cout << nodeName.first << ","<< nodeName.second << " voisin de :\n";
    for(auto n : neighbourhoodNode) {
        NodeName name = n.getName();
        if(name.first != nodeName.first || name.second != nodeName.second) {
        //std::cout << name.first << "," << name.second << " | ";
        addNeighbour(n.getLabel());
        }
    }
   // std::cout <<"\n";
    }
}

NodeName Node::getName() {
    return nodeName;
}

shared_ptr<NodeType> Node::getLabel() {
    return label;
}

vector<shared_ptr<NodeType>> &Node::getNeighbourhood() {
    return neighbourhood;
}

void Node::setNodeType(const NodeType newLabel) {
    if(newLabel == NodeType::NIL) {
        *label = newLabel;
    } else {
        *label = activationVal;
    }

}

void Node::updateNodeState(const State newState) {
    int newIntState = static_cast<int>(newState);
    if(newIntState == 0) {
        *label = NodeType::NIL;
    } else {
        *label = activationVal;
    }
}

State Node::getNodeState() {
    if(*label == NodeType::NIL) {
        return State::DISABLED;
    } else {
        return State::ENABLED;
    }
}

Stats Node::getInteractionStats() {
    Stats stats{};

    if(*label != NodeType::NIL) {
        int label_idx = static_cast<int>(*label);
        int idx{0};

        for(auto const&l : getNeighbourhood()){
            if(*l != NodeType::NIL) {
                idx = label_idx + static_cast<int>(*l) - 2;
                //std::cout << idx << "\n";
                stats.increment(idx);
            }
        }
    }
    return stats;
}

void Node::applyPottsChangeStats(Stats &stats, const State &newState) {
    NodeType newPropType;
    if (newState == State::DISABLED) {
        newPropType = NodeType::NIL;
    } else {
        newPropType = activationVal;
    }

    int labelCurrentIdx = static_cast<int>(*label);
    int labelNewIdx = static_cast<int>(newPropType);

    int idx{0};


    for(int i{0}; i < neighbourhood.size(); i++){
        shared_ptr<NodeType>&l = neighbourhood[i];
        if(*l != NodeType::NIL) {
            if(*label != NodeType::NIL){
                idx = labelCurrentIdx + static_cast<int>(*l) - 2;
                stats.decrement(idx);
            }
            if(newPropType != NodeType::NIL) {
               idx = labelNewIdx + static_cast<int>(*l) - 2;
               stats.increment(idx);
            }
           
        }
       
    }
    *label = newPropType;
}

void Node::getGibbsChangeStatistics(Stats &stats) {
    int idx{0};
    int activation = static_cast<int>(activationVal);
    for(int i{0}; i < neighbourhood.size(); i++){
        shared_ptr<NodeType>&l = neighbourhood[i];
        if(*l != NodeType::NIL) {
            idx = activation + static_cast<int>(*l) - 2;
            stats.increment(idx);
        }
     }
}

