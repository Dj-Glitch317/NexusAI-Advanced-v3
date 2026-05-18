package com.nexusai.assistant.ai.advanced;

import android.content.Context;
import android.util.Log;
import java.util.*;

public class TransferLearningManager {
    private static final String TAG = "TransferLearning-v3";

    private static class DomainKnowledge {
        String name;
        Map<String, Float> facts = new HashMap<>();
        DomainKnowledge(String n) { this.name = n; }
        void addFact(String f, float c) { facts.put(f, c); }
    }

    private final Map<String, DomainKnowledge> domains = new HashMap<>();
    private final Map<String, Float> transferPerformance = new HashMap<>();
    private final float NEGATIVE_THRESHOLD = -0.15f;

    public TransferLearningManager(Context context) {
        initializeCommonSense();
    }

    private void initializeCommonSense() {
        DomainKnowledge cs = new DomainKnowledge("common_sense");
        cs.addFact("sky_is_blue", 0.99f);
        cs.addFact("fire_is_hot", 0.99f);
        domains.put("common_sense", cs);
    }

    public float applyTransfer(String sourceDomain, String targetDomain, float[] weights) {
        float perf = transferPerformance.getOrDefault(sourceDomain, 0.5f);
        if (perf < NEGATIVE_THRESHOLD) {
            Log.w(TAG, "Negative transfer detected for " + sourceDomain + ". Blocking.");
            return 0.0f;
        }
        return perf * 0.2f;
    }

    public void updatePerformance(String domain, float delta) {
        float current = transferPerformance.getOrDefault(domain, 0.0f);
        transferPerformance.put(domain, current * 0.9f + delta * 0.1f);
    }
}
