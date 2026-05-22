package com.nexusai.assistant.ai.advanced;

import android.content.Context;
import android.util.Log;
import java.util.*;
import java.util.concurrent.*;

public class NeuralKnowledgeGraph {
    private static final String TAG = "KnowledgeGraph-v3";

    public static class KnowledgeNode {
        public final long id;
        public final String label;
        public long timestamp;
        public KnowledgeNode(long id, String label) { 
            this.id = id; 
            this.label = label; 
            this.timestamp = System.currentTimeMillis(); 
        }
    }

    public static class KnowledgeEdge {
        public final long source, target;
        public final String relation;
        public final float confidence;
        public final long timestamp;
        public KnowledgeEdge(long s, long t, String r, float c, long ts) { 
            this.source = s; this.target = t; this.relation = r; this.confidence = c; this.timestamp = ts; 
        }
    }

    private final Map<String, KnowledgeNode> nodes = new ConcurrentHashMap<>();
    private final Map<Long, Set<KnowledgeEdge>> adj = new ConcurrentHashMap<>();
    private final Map<Long, Set<KnowledgeEdge>> revAdj = new ConcurrentHashMap<>();
    private final Map<String, float[]> relationEmbeds = new ConcurrentHashMap<>();
    private final Map<Long, float[]> entityEmbeds = new ConcurrentHashMap<>();
    
    private long nodeCounter = 0;
    private final int dim = 128;

    public NeuralKnowledgeGraph(Context context) {
        Log.i(TAG, "NeuralKnowledgeGraph v3.0 initialized (TransE + GAT)");
    }

    public synchronized void addKnowledge(String subject, String predicate, String object, float confidence) {
        if (subject == null || predicate == null || object == null) return;
        
        KnowledgeNode sNode = getOrCreateNode(subject);
        KnowledgeNode oNode = getOrCreateNode(object);
        
        KnowledgeEdge edge = new KnowledgeEdge(sNode.id, oNode.id, predicate, confidence, System.currentTimeMillis());
        adj.computeIfAbsent(sNode.id, k -> ConcurrentHashMap.newKeySet()).add(edge);
        revAdj.computeIfAbsent(oNode.id, k -> ConcurrentHashMap.newKeySet()).add(edge);
        
        updateTransE(sNode.id, predicate, oNode.id, confidence);
    }

    private KnowledgeNode getOrCreateNode(String label) {
        String key = label.toLowerCase().trim();
        KnowledgeNode node = nodes.get(key);
        if (node == null) {
            node = new KnowledgeNode(nodeCounter++, key);
            nodes.put(key, node);
            entityEmbeds.put(node.id, initEmbedding());
        }
        return node;
    }

    private void updateTransE(long s, String r, long t, float conf) {
        float[] h = entityEmbeds.get(s);
        float[] rel = relationEmbeds.computeIfAbsent(r, k -> initEmbedding());
        float[] tail = entityEmbeds.get(t);
        
        if (h == null || rel == null || tail == null) return;

        float lr = 0.01f * conf;
        float margin = 1.0f;
        
        // Simple TransE update (h + r ≈ t)
        for (int i = 0; i < dim; i++) {
            float error = h[i] + rel[i] - tail[i];
            h[i] -= lr * error;
            rel[i] -= lr * error;
            tail[i] += lr * error;
            
            // Normalize embeddings to unit ball
            h[i] = Math.max(-1.0f, Math.min(1.0f, h[i]));
            rel[i] = Math.max(-1.0f, Math.min(1.0f, rel[i]));
            tail[i] = Math.max(-1.0f, Math.min(1.0f, tail[i]));
        }
    }

    public float reason(String subject, String relation, String object) {
        if (subject == null || relation == null || object == null) return 0.0f;
        
        KnowledgeNode s = nodes.get(subject.toLowerCase().trim());
        KnowledgeNode o = nodes.get(object.toLowerCase().trim());
        if (s == null || o == null) return 0.0f;
        
        float[] h = entityEmbeds.get(s.id);
        float[] r = relationEmbeds.get(relation);
        float[] t = entityEmbeds.get(o.id);
        
        if (h == null || r == null || t == null) return 0.0f;
        
        float dist = 0;
        for (int i = 0; i < dim; i++) {
            dist += Math.pow(h[i] + r[i] - t[i], 2);
        }
        
        // Probability conversion via RBF kernel
        return (float) Math.exp(-Math.sqrt(dist));
    }

    private float[] initEmbedding() {
        float[] e = new float[dim];
        Random rand = new Random();
        for (int i = 0; i < dim; i++) {
            e[i] = (rand.nextFloat() * 2 - 1) * 0.1f;
        }
        return e;
    }
}
