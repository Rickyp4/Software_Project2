#include <iostream>
#include <vector>
#include <stdlib.h> 
using namespace std;

class Map{
    public:
    vector<vector<int> > xCol;
    Map(int x, int y){
        vector<int> yEdge;
        for(int i = 0; i < (y + 2); i++){
            yEdge.push_back(1);
        }
        xCol.push_back(yEdge);
        for(int i = 0; i < x; i++){
            vector<int> yVect;
            for(int j = 0; j < (y + 2); j++){
                if (j == 0){
                    yVect.push_back(1);
                }
                else if (j == (y + 1)){
                    yVect.push_back(1);
                }
                else{
                    yVect.push_back(0);
                }
            }
            xCol.push_back(yVect);
        }
        xCol.push_back(yEdge);
        if((rand() % 2)){
            if((rand() % 2)){
                xCol.at(0).at((rand() % y)+1) = -5;
            }
            else{
                xCol.at(x+1).at((rand() % y)+1) = -5;
            }
        }
        else{
            if((rand() % 2)){
                xCol.at((rand() % x)+1).at(0) = -5;
            }
            else{
                xCol.at((rand() % x)+1).at(y+1) = -5;
            }
        }
    }
    
    void printMap(){
        int size1 = xCol.size();
        int size2 = xCol.at(0).size();
        for(int i = 0; i < size2; i++){
            for(int j = 0; j < size1; j++){
                cout << xCol.at(j).at(i) << " ";
            }
            cout << endl;
        }
    }
};

class Goat{
    int x, y, prevX, prevY, facing;
    vector<int> historyX;
    vector<int> historyY;
    
    public:
    Map *m;
    Goat(int x, int y, Map *m){
        this->x = x;
        this->y = y;
        historyX.push_back(x);
        historyY.push_back(y);
        this->m = m;
        m->xCol.at(x).at(y) += 1;
        facing = 0;
    }
    int getPos(char c){
        if(c == 'x'){
            return x;
        }
        else if(c == 'y'){
            return y;
        }
        else{
            return -1;
        }
    }
    void move(){
        if(x == -1){
            return;
        }
        m->xCol.at(x).at(y) -= 1;
        
        prevX = x;
        prevY = y;
        if(facing == 0){
            x++;
        }
        else if(facing == 1){
            y++;
        }
        else if(facing == 2){
            x--;
        }
        else if(facing == 3){
            y--;
        }
        
        m->xCol.at(x).at(y) += 1;
        
        if(m->xCol.at(x).at(y) == 1){
            historyX.push_back(x);
            historyY.push_back(y);
        }
        else if(m->xCol.at(x).at(y) > 1){
            moveBack();
        }
        else{
            historyX.push_back(x);
            historyY.push_back(y);
            m->xCol.at(x).at(y) -= 1;
            
            x = y = -1;
            return;
        }
        
        if((rand() % 2)){
            if(facing == 0){
                facing = 3;
            }
            else{
                facing--;
            }
        }
        else{
            if(facing == 3){
                facing = 0;
            }
            else{
                facing++;
            }
        }
    }
    void moveBack(){
        if(x == -1){
            return;
        }
        m->xCol.at(x).at(y) -= 1;
        
        y = prevY;
        x = prevX;
        historyX.push_back(x);
        historyY.push_back(y);
        
        m->xCol.at(x).at(y) += 1;
    }
    vector<int> getPosHistory(char c){
        if(c == 'x'){
            return historyX;
        }
        else if(c == 'y'){
            return historyY;
        }
        else{
            vector<int> emptyVector;
            return emptyVector;
        }
    }
};

int main()
{
    int input;
    int mapWidth = 20;
    int mapHeight = 20;
    int probability = 50;
    Map *m = new Map(mapWidth, mapHeight);
    vector<Goat*> goats;
    Goat *g;
    for(int i = 0; i < mapWidth; i++){
        for(int j = 0; j < mapHeight; j++){
            if((rand() % 100) < probability){
                g = new Goat((i+1), (j+1), m);
                goats.push_back(g);
            }
        }
    }
    
    m->printMap();
    cout << "Starting map. Press 1 to continue:" << endl << endl;
    //cin >> input;
    input = 1;
    int count = 0;
    while(input == 1){
        input = 0;
        for(int i = 0; i < goats.size(); i++){
            goats.at(i)->move();
            if(goats.at(i)->getPos('x') > -1){
                input = 1;
            }
        }
        //m->printMap();
        count++;
    }
    cout << count << endl;
    return 0;
}
