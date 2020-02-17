#include <iostream>
#include <map>
#include <random>

// get base random alias which is auto seeded and has static API and internal state
//using Random = effolkronium::random_static;
/*
int main()
{
    Random::seed( 10 );
    RandomGen gen{100};
    std::discrete_distribution<> d({40, 10, 10, 40});
    std::map<int, int> m;
    for(int n=0; n<10000; ++n) {
        ++m[d(gen.getRandomLocal().engine())];
    }
    for(auto p : m) {
        std::cout << p.first << " generated " << p.second << " times\n";
    }
}*/
