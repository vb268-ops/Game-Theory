#include<iostream>
#include <algorithm>
#include <vector>
#include<cmath> // For atan
#include<random> // For the random point

using namespace std;

// --- Function ---
float distanceCalculate(float x1, float y1, float x2, float y2)
{
	float x = x1 - x2; //calculating number to square in next step
	float y = y1 - y2;
	float dist;

	dist = pow(x, 2) + pow(y, 2); //calculating Euclidean distance
	dist = sqrt(dist);                  

    // cout<<" Eucledian Distance: "<<dist<<"\n";

	return dist;
}

float normalized_value(float max_new, float min_new, float max_old, float min_old, float query_value)
{
    float normalized_value;

    float element1 = max_new-min_new, element2 = max_old - min_old;
    float element3 = query_value - max_old, element4 = max_new;

    normalized_value = ((element1/element2)*element3) + element4;

    return normalized_value;
}

// --- Function ---
float* basis_function_values(float expert_rollout[][2], float learner_rollout[][2], int number_of_waypoints, float max_new, float min_new)
{
    // --- Storing values to find the maximum for waypt 2 waypt lengths ---
    vector<float> expert_waypt2waypt_length, expert_waypt2waypt_slope, learner_waypt2waypt_length, learner_waypt2waypt_slope; 
    for(int i=0; i<number_of_waypoints-1; i++)
    {
        expert_waypt2waypt_length.push_back(distanceCalculate(expert_rollout[i][0], expert_rollout[i][1], expert_rollout[i+1][0], expert_rollout[i+1][1]));
        expert_waypt2waypt_slope.push_back(atan((expert_rollout[i+1][1]-expert_rollout[i][1])/(expert_rollout[i+1][0]-expert_rollout[i][0])));

        learner_waypt2waypt_length.push_back(distanceCalculate(learner_rollout[i][0], learner_rollout[i][1], learner_rollout[i+1][0], learner_rollout[i+1][1]));
        learner_waypt2waypt_slope.push_back(atan((learner_rollout[i+1][1]-learner_rollout[i][1])/(learner_rollout[i+1][0]-learner_rollout[i][0])));
    }

    float max_expert_waypt2waypt_length = *max_element(expert_waypt2waypt_length.begin(), expert_waypt2waypt_length.end()), min_expert_waypt2waypt_length = *min_element(expert_waypt2waypt_length.begin(), expert_waypt2waypt_length.end());
    float max_expert_waypt2waypt_slope = *max_element(expert_waypt2waypt_slope.begin(), expert_waypt2waypt_slope.end()), min_expert_waypt2waypt_slope = *min_element(expert_waypt2waypt_slope.begin(), expert_waypt2waypt_slope.end());

    float max_learner_waypt2waypt_length = *max_element(learner_waypt2waypt_length.begin(), learner_waypt2waypt_length.end()), min_learner_waypt2waypt_length = *min_element(learner_waypt2waypt_length.begin(), learner_waypt2waypt_length.end());
    float max_learner_waypt2waypt_slope = *max_element(learner_waypt2waypt_slope.begin(), learner_waypt2waypt_slope.end()), min_learner_waypt2waypt_slope = *min_element(learner_waypt2waypt_slope.begin(), learner_waypt2waypt_slope.end());

    // --- Normalized length and slope ---
    float expert_length = 0, expert_slope = 0, learner_length = 0, learner_slope = 0;
    for(int i=0; i<number_of_waypoints-1; i++)
    {   
        expert_length = expert_length + normalized_value(max_new, min_new, max_expert_waypt2waypt_length, min_expert_waypt2waypt_length, expert_waypt2waypt_length[i]);
        expert_slope = expert_slope + normalized_value(max_new, min_new, max_expert_waypt2waypt_slope, min_expert_waypt2waypt_slope, expert_waypt2waypt_slope[i]);

        learner_length = learner_length + normalized_value(max_new, min_new, max_learner_waypt2waypt_length, min_learner_waypt2waypt_length, learner_waypt2waypt_length[i]);
        learner_slope = learner_slope + normalized_value(max_new, min_new, max_learner_waypt2waypt_slope, min_learner_waypt2waypt_slope, learner_waypt2waypt_slope[i]);
    }

    cout<<"EL: "<<expert_length<<" ES: "<<expert_slope<<" LL: "<<learner_length<<" LS: "<<learner_slope<<"\n";

    // --- Final Declarations ---
    static float basis_function_val[4];
    basis_function_val[0] = expert_length; basis_function_val[1] = expert_slope; basis_function_val[2] = learner_length; basis_function_val[3] = learner_slope;

    return basis_function_val;
}

// --- Function ---
float* discriminator(float learner_length, float learner_slope, float w1_plus_w2)
{
    // Taking w1+w2 = 100

    float w1=0, w2;

    static float w[2];

    float delta_w1 = 0.05*w1_plus_w2; // The increment/decrement in w1/w2.

    float cost = -1;

    while (w1!=w1_plus_w2)
    {
        w2 = w1_plus_w2 - w1;
        float intermediate_cost = w1*learner_length + w2*learner_slope;

        if (intermediate_cost>cost)
        {
            cost = intermediate_cost;
            w[0] = w1; w[1] = w2;
        }
        w1 = w1 + delta_w1;
    } 

    return w;
}