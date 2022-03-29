#include <vector>
#include <iostream>

using namespace std;

struct Connection
{
    double weight;
    double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

// <***************** class Neuron *****************>

class Neuron
{
    public:
        Neuron(unsigned numOutputs);

    private:
        double m_outputVal;
        vector<Connection> m_outputWeights;  
};

Neuron::Neuron(unsigned numOutputs)
{
    for (unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
    }
    
}

// <***************** class Net *****************>

class Net
{
    public:
        Net(const vector<unsigned> &topology);
        void feedForward(const vector<double> &inputVals) {};
        void backProp(const vector<double> &targetVals) {};
        void getResults(vector<double> &resultVals) const {};


    private: 
        vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
};
 
Net::Net(const vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        // make new layer
        m_layers.push_back(Layer());
        unsigned numOutputs == topology.size() -1 ? 0 : topology[layerNum + 1];

        // add a bias neuron to the layer
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum){
            m_layers.back().push_back(Neuron(numOutputs));
            cout << "Made a Neuron" << endl;
        }
    }
}

int main()
{
    // e.g., (3, 2, 1)
    vector<unsigned> topology;
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);
    Net myNet(topology);
    
    vector<double> inputVals;
    myNet.feedForward(inputVals);

    vector<double> targetVals;
    myNet.backProp(targetVals);

    vector<double> resultVals;
    myNet.getResults(resultVals);
}