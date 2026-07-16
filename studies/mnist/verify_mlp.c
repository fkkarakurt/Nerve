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

/* Verify the exported mnist_mlp.bin + the exact forward the browser uses
 * (ReLU hidden + softmax) reproduce the training accuracy on the test set. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int rd(FILE*f){unsigned char b[4];if(fread(b,1,4,f)!=4)return -1;return (b[0]<<24)|(b[1]<<16)|(b[2]<<8)|b[3];}

int main(void)
{
    FILE *w=fopen("mnist_mlp.bin","rb"); int nin,nhid,nout,i,j,k;
    if(!w){printf("no mnist_mlp.bin\n");return 1;}
    fread(&nin,4,1,w);fread(&nhid,4,1,w);fread(&nout,4,1,w);
    float *W1=malloc((size_t)nhid*nin*4),*b1=malloc((size_t)nhid*4),*W2=malloc((size_t)nout*nhid*4),*b2=malloc((size_t)nout*4);
    fread(W1,4,(size_t)nhid*nin,w);fread(b1,4,(size_t)nhid,w);fread(W2,4,(size_t)nout*nhid,w);fread(b2,4,(size_t)nout,w);fclose(w);

    FILE *fi=fopen("data/t10k-images-idx3-ubyte","rb"),*fl=fopen("data/t10k-labels-idx1-ubyte","rb");
    if(!fi||!fl){printf("no test data\n");return 1;}
    rd(fi);int n=rd(fi);rd(fi);rd(fi); rd(fl);rd(fl);
    unsigned char *img=malloc((size_t)nin),*lab=malloc((size_t)n);
    fread(lab,1,(size_t)n,fl);fclose(fl);
    float *px=malloc((size_t)nin*4),*h=malloc((size_t)nhid*4),*o=malloc((size_t)nout*4);
    int correct=0;
    for(int s=0;s<n;s++){
        fread(img,1,(size_t)nin,fi);
        for(i=0;i<nin;i++)px[i]=img[i]/255.0f;
        for(j=0;j<nhid;j++){float v=b1[j];for(i=0;i<nin;i++)v+=W1[j*nin+i]*px[i];h[j]=v>0?v:0;}
        for(k=0;k<nout;k++){float v=b2[k];for(j=0;j<nhid;j++)v+=W2[k*nhid+j]*h[j];o[k]=v;}
        int best=0;for(k=1;k<nout;k++)if(o[k]>o[best])best=k;
        if(best==lab[s])correct++;
    }
    fclose(fi);
    printf("forward-from-export accuracy: %.2f%% (%d/%d)  [arch %d-%d-%d]\n",
           100.0*correct/n,correct,n,nin,nhid,nout);
    return 0;
}
