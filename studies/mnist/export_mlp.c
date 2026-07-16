/*
 * Copyright 2022-2026 Fatih Kucukkarakurt <fatihkucukkarakurt@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * export_mlp.c — train the 784-128-10 MNIST MLP and export its weights as a
 * flat little-endian file the browser can load: the digit recogniser that
 * powers the "draw a number" demo.
 *
 * Format (mnist_mlp.bin):
 *   int32 n_in, n_hid, n_out
 *   float W1[n_hid*n_in], b1[n_hid]     (hidden, ReLU)
 *   float W2[n_out*n_hid], b2[n_out]    (output, softmax)
 *
 * Build:  gcc -O3 -march=native export_mlp.c -o export_mlp -lm
 * Run:    ./export_mlp [epochs] [train_cap]
 */
#define NERVE_IMPLEMENTATION
#include "../../nerve.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IMG 784
#define CLS 10
#define HID 128

static int read_be32(FILE *f)
{
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return -1;
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}

/* IDX image file: magic 2051, count, rows, cols, then one byte per pixel. */
static int load_images(const char *p, float **o)
{
    FILE *f = fopen(p, "rb");
    int m, n, r, c, i;
    unsigned char *b;
    float *d;

    if (!f) return -1;
    m = read_be32(f); n = read_be32(f); r = read_be32(f); c = read_be32(f);
    if (m != 2051 || r * c != IMG) { fclose(f); return -1; }

    b = (unsigned char *)malloc((size_t)n * IMG);
    d = (float *)malloc((size_t)n * IMG * sizeof(float));
    if (!b || !d || fread(b, 1, (size_t)n * IMG, f) != (size_t)n * IMG) {
        free(b); free(d); fclose(f); return -1;
    }
    fclose(f);

    for (i = 0; i < n * IMG; i++) d[i] = b[i] / 255.0f;
    free(b);
    *o = d;
    return n;
}

/* IDX label file: magic 2049, count, then one byte per label. Returned
 * one-hot, which is the shape net_train_epoch expects. */
static int load_labels(const char *p, float **o)
{
    FILE *f = fopen(p, "rb");
    int m, n, i;
    unsigned char *b;
    float *d;

    if (!f) return -1;
    m = read_be32(f); n = read_be32(f);
    if (m != 2049) { fclose(f); return -1; }

    b = (unsigned char *)malloc((size_t)n);
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) {
        free(b); fclose(f); return -1;
    }
    fclose(f);

    d = (float *)calloc((size_t)n * CLS, sizeof(float));
    if (!d) { free(b); return -1; }
    for (i = 0; i < n; i++) d[i * CLS + b[i]] = 1.0f;
    free(b);
    *o = d;
    return n;
}

int main(int argc,char**argv)
{
    int epochs=(argc>1)?atoi(argv[1]):3, cap=(argc>2)?atoi(argv[2]):60000, ntr,nte,e,nu,nl;
    float *Xtr,*Ytr,*Xte,*Yte; FILE*o;
    ntr=load_images("data/train-images-idx3-ubyte",&Xtr);
    nte=load_images("data/t10k-images-idx3-ubyte",&Xte);
    if(load_labels("data/train-labels-idx1-ubyte",&Ytr)!=ntr) return 1;
    if(load_labels("data/t10k-labels-idx1-ubyte",&Yte)!=nte) return 1;
    if(cap<ntr) ntr=cap;
    printf("train %d test %d epochs %d\n",ntr,nte,epochs);

    nerve_seed(42);
    network_t*net=net_allocate(3,IMG,HID,CLS);
    net_set_optimizer(net,NERVENET_OPTIMIZER_ADAM);
    net_set_activation(net,NERVENET_ACTIVATION_RELU);
    net_set_classification(net);
    net_initialize_he(net);
    net_set_learning_rate(net,0.001f);
    net_set_l2_lambda(net,1e-5f);
    for(e=1;e<=epochs;e++){
        float loss=net_train_epoch(net,Xtr,Ytr,ntr,IMG,CLS,1);
        float acc=net_compute_accuracy(net,Xte,Yte,nte,IMG,CLS);
        printf("  epoch %d  loss %.4f  test %.2f%%\n",e,loss,100.0f*acc); fflush(stdout);
    }

    /* export flat weights */
    o=fopen("mnist_mlp.bin","wb");
    { int hin=IMG,hh=HID,ho=CLS; fwrite(&hin,4,1,o); fwrite(&hh,4,1,o); fwrite(&ho,4,1,o); }
    /* hidden layer (net->layer[1]): per neuron weights over IMG inputs */
    for(nu=0;nu<HID;nu++) for(nl=0;nl<IMG;nl++) fwrite(&net->layer[1].neuron[nu].weight[nl],4,1,o);
    for(nu=0;nu<HID;nu++) fwrite(&net->layer[1].neuron[nu].weight[IMG],4,1,o);   /* bias */
    /* output layer (net->layer[2]): per neuron weights over HID hidden */
    for(nu=0;nu<CLS;nu++) for(nl=0;nl<HID;nl++) fwrite(&net->layer[2].neuron[nu].weight[nl],4,1,o);
    for(nu=0;nu<CLS;nu++) fwrite(&net->layer[2].neuron[nu].weight[HID],4,1,o);   /* bias */
    fclose(o);
    printf("wrote mnist_mlp.bin\n");
    net_free(net); free(Xtr);free(Ytr);free(Xte);free(Yte);
    return 0;
}
