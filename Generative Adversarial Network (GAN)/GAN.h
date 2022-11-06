#include<iostream>
#include<stdio.h>
#include<random>
#include <ctime>
#include <cstdlib>

using namespace std;

// The Generator
double* generator_inference(double g_w1, double g_w2, int noise)
{
    static double G_z[2];

    G_z[0] = exp(g_w1*noise)/(1 + exp(g_w1*noise)); 
    G_z[1] = exp(g_w2*noise)/(1 + exp(g_w2*noise));

    return G_z;
}

// The Discriminator
double discriminator_inference(double d_w1, double d_w2, double* G_z)
{
    double D_G_z;

    D_G_z = exp(d_w1*G_z[0] + d_w1*G_z[1])/(1 + exp(d_w1*G_z[0] + d_w1*G_z[1]));

    return D_G_z;
}

// Training Generator and Discriminator
double* training(double g_w1, double g_w2, double d_w1, double d_w2, double* G_z, double D_G_z, int noise, double lr)
{
    // Formula for Gradient for GD taken from https://www.youtube.com/watch?v=8L11aMN5KY8
    static double updated_weights[4];

    // Generator
    g_w1 = g_w1 - lr*(-(1 - D_G_z)*G_z[0]*(1-G_z[0])*noise);
    g_w2 = g_w2 - lr*(-(1 - D_G_z)*G_z[1]*(1-G_z[1])*noise);

    // Discriminator
    d_w1 = d_w1 - lr*(-(1 - D_G_z)); 
    d_w2 = d_w2 - lr*(-(1 - D_G_z));

    updated_weights[0] = g_w1; updated_weights[1] = g_w2; updated_weights[2] = d_w1; updated_weights[3] = d_w2;

    return updated_weights; 
}