package com.nexusai.assistant.ai.advanced;

import android.content.Context;
import java.util.*;

public class ActiveLearningSystem {
    public enum SamplingStrategy { BALD, CORE_SET, HYBRID, UNCERTAINTY }
    private SamplingStrategy strategy = SamplingStrategy.HYBRID;
    private final Map<String, float[]> featureStore = new HashMap<>();

    public float calculateImportance(AdvancedLearningEngine.LearningExample ex) {
        if (ex == null || ex.inputFeatures == null) return 0.0f;
        
        featureStore.put(String.valueOf(ex.id), ex.inputFeatures);
        switch (strategy) {
            case BALD: return calculateBALD(ex);
            case CORE_SET: return calculateCoreSet(ex);
            case UNCERTAINTY: return 1.0f - ex.confidence;
            default: return calculateHybrid(ex);
        }
    }

    private float calculateBALD(AdvancedLearningEngine.LearningExample ex) {
        // Principal entropy - Aleatoric uncertainty
        float p = Math.max(1e-6f, Math.min(1.0f - 1e-6f, ex.confidence));
        float entropy = (float) -(p * Math.log(p) + (1-p) * Math.log(1-p));
        float aleatoric = 0.1f; // Simulated base uncertainty
        return Math.max(0, entropy - aleatoric);
    }

    private float calculateCoreSet(AdvancedLearningEngine.LearningExample ex) {
        if (featureStore.size() <= 1) return 1.0f;
        float minEntryDist = Float.MAX_VALUE;
        for (Map.Entry<String, float[]> entry : featureStore.entrySet()) {
            if (entry.getKey().equals(String.valueOf(ex.id))) continue;
            float dist = cosineDistance(ex.inputFeatures, entry.getValue());
            minEntryDist = Math.min(minEntryDist, dist);
        }
        return minEntryDist; // Higher distance means more unique/important for core set
    }

    private float calculateHybrid(AdvancedLearningEngine.LearningExample ex) {
        return (calculateBALD(ex) * 0.4f + calculateCoreSet(ex) * 0.6f);
    }

    private float cosineDistance(float[] a, float[] b) {
        if (a == null || b == null || a.length != b.length) return 1.0f;
        float dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        float denominator = (float) (Math.sqrt(normA) * Math.sqrt(normB));
        if (denominator < 1e-8f) return 1.0f;
        return 1.0f - (dot / denominator);
    }

    public void setStrategy(SamplingStrategy strategy) {
        this.strategy = strategy;
    }
}
