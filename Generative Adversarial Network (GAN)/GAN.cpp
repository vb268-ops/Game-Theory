// A GAN being trained to take noise input and output values within certain Gaussian Distribution.
// On the lines of discussion by "https://www.youtube.com/watch?v=8L11aMN5KY8"

#include<iostream>
#include<stdio.h>
#include<random>
#include<ctime>
#include<cstdlib>
#include"GAN.h"

using namespace std;

int main(){

    // Initial setting to draw values from Gaussian
    srand(time(0));

    // Initial weight declaration
    double g_w1 = 0.01, g_w2 = 0.01, d_w1 = 0.01, d_w2 = 0.01;

    // Final weights that would be stored
    double final_g_w1, final_g_w2, final_d_w1, final_d_w2, loss=1000;

    // Declaring the True Gaussian
    int mean_x = 25, std_x = 15;
    int mean_y = 25, std_y = 15;

    int number_of_iterations = 7500;

    for(int i=0; i<number_of_iterations; i++)
    {
        // Real value sampled from the GT Gaussian
        double true_x = (mean_x - std_x) + (rand() % (2*std_x));
        double true_y = (mean_y - std_y) + (rand() % (2*std_y));

        // Generating Noise
        int noise_lower_bound = 0, noise_upper_bound = 100;
        int noise = noise_lower_bound + (rand() % (noise_upper_bound - noise_lower_bound));

        // THE GAME
        // 1. Inference
        double* G_z;
        G_z = generator_inference(g_w1, g_w2, noise);
        double D_G_z;
        D_G_z = discriminator_inference(d_w1, d_w2, G_z);

        // 2. Loss (using log-loss as per https://www.youtube.com/watch?v=8L11aMN5KY8)
        double generator_loss = -log(D_G_z), discriminator_loss = -log(1 - D_G_z);

        // 3. Training to update weights
        double learning_rate = 0.001;
        double* updated_weights;
        updated_weights = training(g_w1, g_w2, d_w1, d_w2, G_z, D_G_z, noise, learning_rate);

        g_w1 = updated_weights[0], g_w2 = updated_weights[1], d_w1 = updated_weights[2], d_w2 = updated_weights[3];

        // Storing weights for case with lowest loss
        if (generator_loss<loss)
        {
            final_g_w1 = g_w1; final_g_w2 = g_w2; final_d_w1 = d_w1; final_d_w2 = d_w2; loss = generator_loss;
        }

        cout<<"Noise: "<<noise<<" "<<"Infer.: "<<G_z[0]<<" "<<G_z[1]<<" Gen. Loss: "<<generator_loss<<" Dis. Loss: "<<discriminator_loss<<"\n";
    }

    cout<<" Final Loss: "<<loss<<"\n";

    return 0; 

}
