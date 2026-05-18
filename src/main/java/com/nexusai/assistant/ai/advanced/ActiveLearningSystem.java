package com.nexusai.assistant.ai.advanced;

import android.content.Context;
import java.util.*;

public class ActiveLearningSystem {
    public enum SamplingStrategy { BALD, CORE_SET, HYBRID, UNCERTAINTY }
    private SamplingStrategy strategy = SamplingStrategy.HYBRID;
    private final Map<String, float[]> featureStore = new HashMap<>();

    public float calculateImportance(AdvancedLearningEngine.LearningExample ex) {
        featureStore.put(String.valueOf(ex.id), ex.inputFeatures);
        switch (strategy) {
            case BALD: return calculateBALD(ex);
            case CORE_SET: return calculateCoreSet(ex);
            default: return calculateHybrid(ex);
        }
    }

    private float calculateBALD(AdvancedLearningEngine.LearningExample ex) {
        float entropy = (float) Math.log(ex.confidence + 1e-6);
        float aleatoric = 0.1f;
        return Math.abs(entropy - aleatoric);
    }

    private float calculateCoreSet(AdvancedLearningEngine.LearningExample ex) {
        if (featureStore.isEmpty()) return 1.0f;
        float maxMinDist = 0;
        for (float[] feat : featureStore.values()) {
            float dist = cosineDistance(ex.inputFeatures, feat);
            maxMinDist = Math.max(maxMinDist, dist);
        }
        return maxMinDist;
    }

    private float calculateHybrid(AdvancedLearningEngine.LearningExample ex) {
        return (calculateBALD(ex) + calculateCoreSet(ex)) / 2.0f;
    }

    private float cosineDistance(float[] a, float[] b) {
        float dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return 1.0f - (dot / (float) (Math.sqrt(normA) * Math.sqrt(normB) + 1e-6));
    }
}
