package com.nexusai.assistant.ai.advanced;

import android.content.Context;
import android.util.Log;
import java.util.*;

public class TransformerAttention {
    private static final String TAG = "TransformerAttention-v3";

    private final int modelDimension;
    private final int numAttentionHeads;
    private final int numKVHeads;
    private final int headDimension;
    private final int numLayers;
    private final int maxSequenceLength;
    private final int feedForwardDimension;

    private float[][][] queryWeights; // [layer][dim][dim]
    private float[][][] keyWeights;
    private float[][][] valueWeights;
    private float[][][] outputProjection;
    
    private float[][] rmsNormGamma;
    private float[][] rmsNormGammaFF;
    private float[][] layerScaleAttention;
    private float[][] layerScaleFF;

    private float[][][] ffW1, ffW2, ffW3;

    private float[][] ropeCos, ropeSin;
    private final float ropeBase = 10000.0f;

    public TransformerAttention(Context context, int modelDim, int numHeads, int numKVH, int layers, int maxSeqLen) {
        this.modelDimension = modelDim;
        this.numAttentionHeads = numHeads;
        this.numKVHeads = Math.min(numKVH, numHeads);
        this.headDimension = modelDim / numHeads;
        this.numLayers = layers;
        this.maxSequenceLength = maxSeqLen;
        this.feedForwardDimension = modelDim * 4;

        initializeParameters();
        initializeRoPE();
    }

    private void initializeParameters() {
        queryWeights = new float[numLayers][modelDimension][modelDimension];
        keyWeights = new float[numLayers][modelDimension][modelDimension];
        valueWeights = new float[numLayers][modelDimension][modelDimension];
        outputProjection = new float[numLayers][modelDimension][modelDimension];
        
        rmsNormGamma = new float[numLayers][modelDimension];
        rmsNormGammaFF = new float[numLayers][modelDimension];
        layerScaleAttention = new float[numLayers][modelDimension];
        layerScaleFF = new float[numLayers][modelDimension];

        ffW1 = new float[numLayers][modelDimension][feedForwardDimension];
        ffW3 = new float[numLayers][modelDimension][feedForwardDimension];
        ffW2 = new float[numLayers][feedForwardDimension][modelDimension];
        
        for (int l = 0; l < numLayers; l++) {
            fillRandom(queryWeights[l]);
            fillRandom(keyWeights[l]);
            fillRandom(valueWeights[l]);
            fillRandom(outputProjection[l]);
            
            Arrays.fill(rmsNormGamma[l], 1.0f);
            Arrays.fill(rmsNormGammaFF[l], 1.0f);
            Arrays.fill(layerScaleAttention[l], 0.02f);
            Arrays.fill(layerScaleFF[l], 0.02f);

            fillRandom(ffW1[l]);
            fillRandom(ffW3[l]);
            fillRandom(ffW2[l]);
        }
    }

    private void initializeRoPE() {
        ropeCos = new float[maxSequenceLength][headDimension / 2];
        ropeSin = new float[maxSequenceLength][headDimension / 2];
        for (int i = 0; i < maxSequenceLength; i++) {
            for (int j = 0; j < headDimension / 2; j++) {
                float theta = (float) (i * Math.pow(ropeBase, -2.0 * j / headDimension));
                ropeCos[i][j] = (float) Math.cos(theta);
                ropeSin[i][j] = (float) Math.sin(theta);
            }
        }
    }

    private void fillRandom(float[][] matrix) {
        Random rand = new Random();
        float scale = (float) Math.sqrt(2.0 / (matrix.length + matrix[0].length));
        for (float[] row : matrix) {
            for (int i = 0; i < row.length; i++) {
                row[i] = (rand.nextFloat() * 2 - 1) * scale;
            }
        }
    }

    public float[] forward(float[] input) {
        if (input == null || input.length != modelDimension) return new float[modelDimension];
        
        float[] x = input;
        for (int l = 0; l < numLayers; l++) {
            float[] xNorm = applyRMSNorm(x, rmsNormGamma[l]);
            float[] attn = computeAttention(xNorm, l);
            x = addAndScale(x, attn, layerScaleAttention[l]);
            
            float[] ffNorm = applyRMSNorm(x, rmsNormGammaFF[l]);
            float[] ffn = computeSwiGLU(ffNorm, l);
            x = addAndScale(x, ffn, layerScaleFF[l]);
        }
        return x;
    }

    private float[] applyRMSNorm(float[] x, float[] gamma) {
        float sumSq = 0;
        for (float v : x) sumSq += v * v;
        float invRms = (float) (1.0 / Math.sqrt(sumSq / x.length + 1e-6));
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++) out[i] = x[i] * invRms * gamma[i];
        return out;
    }

    private float[] computeAttention(float[] x, int l) {
        float[] q = multiply(x, queryWeights[l]);
        float[] k = multiply(x, keyWeights[l]);
        float[] v = multiply(x, valueWeights[l]);

        // Multi-head split and RoPE
        for (int h = 0; h < numAttentionHeads; h++) {
            applyRoPEToHead(q, h, 0);
            applyRoPEToHead(k, h, 0);
        }

        float[] headOutputs = new float[modelDimension];
        for (int h = 0; h < numAttentionHeads; h++) {
            float score = 0;
            int offset = h * headDimension;
            for (int i = 0; i < headDimension; i++) {
                score += q[offset + i] * k[offset + i];
            }
            score /= (float) Math.sqrt(headDimension);
            float prob = (float) Math.exp(score); // Simplified softmax for single token

            for (int i = 0; i < headDimension; i++) {
                headOutputs[offset + i] = prob * v[offset + i];
            }
        }

        return multiply(headOutputs, outputProjection[l]);
    }

    private void applyRoPEToHead(float[] vec, int headIdx, int pos) {
        int offset = headIdx * headDimension;
        for (int i = 0; i < headDimension / 2; i++) {
            float x1 = vec[offset + i];
            float x2 = vec[offset + i + headDimension / 2];
            float cos = ropeCos[pos][i];
            float sin = ropeSin[pos][i];
            vec[offset + i] = x1 * cos - x2 * sin;
            vec[offset + i + headDimension / 2] = x1 * sin + x2 * cos;
        }
    }

    private float[] computeSwiGLU(float[] x, int l) {
        float[] x1 = multiply(x, ffW1[l]);
        float[] x3 = multiply(x, ffW3[l]);
        
        for (int i = 0; i < x3.length; i++) {
            float silu = (float) (x3[i] * (1.0 / (1.0 + Math.exp(-x3[i]))));
            x1[i] *= silu;
        }
        
        return multiply(x1, ffW2[l]);
    }

    private float[] multiply(float[] v, float[][] m) {
        float[] out = new float[m[0].length];
        for (int i = 0; i < v.length; i++) {
            for (int j = 0; j < m[0].length; j++) {
                out[j] += v[i] * m[i][j];
            }
        }
        return out;
    }

    private float[] addAndScale(float[] a, float[] b, float[] scale) {
        float[] out = new float[a.length];
        for (int i = 0; i < a.length; i++) out[i] = a[i] + b[i] * scale[i];
        return out;
    }
}
