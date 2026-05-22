package com.nexusai.assistant.ai.advanced;

import android.content.Context;
import android.util.Log;
import java.util.*;
import java.util.concurrent.*;

public class TransferLearningManager {
    private static final String TAG = "TransferLearning-v3";

    private static class DomainKnowledge {
        String name;
        Map<String, Float> facts = new ConcurrentHashMap<>();
        DomainKnowledge(String n) { this.name = n; }
        void addFact(String f, float c) { facts.put(f, c); }
    }

    private final Map<String, DomainKnowledge> domains = new ConcurrentHashMap<>();
    private final Map<String, Float> transferPerformance = new ConcurrentHashMap<>();
    private final float NEGATIVE_THRESHOLD = -0.3f;

    public TransferLearningManager(Context context) {
        initializeCommonSense();
    }

    private void initializeCommonSense() {
        DomainKnowledge cs = new DomainKnowledge("common_sense");
        cs.addFact("sky_is_blue", 0.99f);
        cs.addFact("fire_is_hot", 0.99f);
        cs.addFact("logic_exists", 1.0f);
        domains.put("common_sense", cs);
        
        DomainKnowledge coding = new DomainKnowledge("coding");
        coding.addFact("null_safety_important", 0.95f);
        coding.addFact("complexity_affects_perf", 0.98f);
        domains.put("coding", coding);
    }

    public float applyTransfer(String sourceDomain, String targetDomain, float[] inputFeatures) {
        if (sourceDomain == null || targetDomain == null) return 0.0f;
        
        float perf = transferPerformance.getOrDefault(sourceDomain, 0.5f);
        if (perf < NEGATIVE_THRESHOLD) {
            Log.w(TAG, "Critical negative transfer in " + sourceDomain + ". Blocking knowledge leak.");
            return 0.0f;
        }

        // Higher performance and matching domains yield better transfer coefficients
        float domainSimilarity = sourceDomain.equals(targetDomain) ? 1.0f : 0.4f;
        return Math.max(0, perf * domainSimilarity * 0.5f);
    }

    public void updatePerformance(String domain, float delta) {
        if (domain == null) return;
        float current = transferPerformance.getOrDefault(domain, 0.5f);
        // Exponential moving average for stability
        transferPerformance.put(domain, Math.max(-1.0f, Math.min(1.0f, current * 0.85f + delta * 0.15f)));
    }
}
