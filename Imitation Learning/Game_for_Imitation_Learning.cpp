// A Game-Theoretic framework for a straight line path using Imitation Learning.
// Vaibhav Bisht (MS, Mechanical Engineering, Cornell University)

#include<iostream>
#include<cmath> // For atan
#include<random> // For the random point

#include"Game_for_Imitation_learning.h"

using namespace std;

int main()
{
    // Declaring expert rollout
    float expert_rollout[9][2] = {{0,0}, {10,0}, {20, 0}, {30, 0}, {40,0}, {60,0}, {60,20}, {60,40}, {60,60}}; // Expert Demonstration
    
    const int number_of_waypoints = sizeof(expert_rollout)/sizeof(expert_rollout[0]); // Number of waypoints in expert rollout (same to be used in learner)
    float radius = 5; // Search radius for the random trajectory
    int number_of_iterations = 25; // Number of iterations of the game

    float epsilon = 500;

    for(int flag = 0; flag<number_of_iterations; flag++) // Number of iterations of the entire procedure
    {
        // Creating learner trajectory for this iteration
        // ToDo: make an array of entities that are not allowed
        float learner_trajectory[number_of_waypoints][2];
        for(int i=0; i<number_of_waypoints; i++)
        {
            learner_trajectory[i][0] = (expert_rollout[i][0]-radius) + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/((2*radius))));
            learner_trajectory[i][1] = (expert_rollout[i][1]-radius) + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/((2*radius))));
        }

        // Length & slope of expert and learner
        float* basis_function_val;
        float max_new = 100, min_new = 50;
        basis_function_val = basis_function_values(expert_rollout, learner_trajectory, number_of_waypoints, max_new, min_new);
        float expert_length = basis_function_val[0], expert_slope=basis_function_val[1], learner_length=basis_function_val[2], learner_slope=basis_function_val[3];

        // Discriminator calculating weights to criticise basis function
        float *w;
        float w1, w2, w1_plus_w2 = 100;
        w = discriminator(learner_length, learner_slope, w1_plus_w2);
        w1 = w[0]; w2 = w[1];

        // Perfoming final min_max operation of Game Theory
        float expert_cost = w1*expert_length + w2*expert_slope;
        float learner_cost = w1*learner_length + w2*learner_slope;

        cout<<"*** Iter. No.: "<<flag<<" w1: "<<w1<<" "<<" w2: "<<w2<<" EC: "<<expert_cost<<" LC: "<<learner_cost<<" *** \n";

        if (abs(learner_cost-expert_cost) < epsilon)
        {
            cout<<" Converged!"<<"\n";
            cout<<" Length: "<<learner_length<<" "<<" Slope: "<<learner_slope<<"\n";
            cout<<" Delta Cost "<<abs(learner_cost-expert_cost)<<"\n";
            break;
        }
    }

    return 0;
}
