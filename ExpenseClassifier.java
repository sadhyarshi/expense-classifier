import java.util.*;

public class ExpenseClassifier {

    // category -> (word -> count)
    private final Map<String, Map<String, Integer>> wordCounts = new HashMap<>();
    // category -> total words seen
    private final Map<String, Integer> totalWords = new HashMap<>();
    // category -> number of training docs
    private final Map<String, Integer> docCounts = new HashMap<>();
    private final Set<String> vocab = new HashSet<>();
    private int totalDocs = 0;

    static List<String> tokenize(String text) {
        List<String> out = new ArrayList<>();
        for (String w : text.toLowerCase().split("[^a-z]+")) {
            if (w.length() > 1) out.add(w);   // drops numbers and 1-letter noise
        }
        return out;
    }

    void train(String category, String text) {
        category = category.toLowerCase().trim();
        docCounts.merge(category, 1, Integer::sum);
        totalDocs++;
        Map<String, Integer> counts = wordCounts.computeIfAbsent(category, k -> new HashMap<>());
        for (String w : tokenize(text)) {
            counts.merge(w, 1, Integer::sum);
            totalWords.merge(category, 1, Integer::sum);
            vocab.add(w);
        }
    }

    /** Returns categories sorted by probability (best first). */
    List<Map.Entry<String, Double>> predict(String text) {
        List<String> words = tokenize(text);
        Map<String, Double> logScores = new HashMap<>();

        for (String cat : docCounts.keySet()) {
            double score = Math.log((double) docCounts.get(cat) / totalDocs);   // prior
            Map<String, Integer> counts = wordCounts.get(cat);
            int total = totalWords.getOrDefault(cat, 0);
            for (String w : words) {
                if (!vocab.contains(w)) continue;   // ignore words never seen in training
                // Laplace smoothing so unseen (cat, word) pairs don't give probability 0
                double p = (counts.getOrDefault(w, 0) + 1.0) / (total + vocab.size());
                score += Math.log(p);
            }
            logScores.put(cat, score);
        }

        // Convert log scores to probabilities (softmax, shifted for numeric stability)
        double max = Collections.max(logScores.values());
        double sum = 0;
        Map<String, Double> probs = new HashMap<>();
        for (var e : logScores.entrySet()) {
            double v = Math.exp(e.getValue() - max);
            probs.put(e.getKey(), v);
            sum += v;
        }
        List<Map.Entry<String, Double>> ranked = new ArrayList<>();
        for (var e : probs.entrySet()) ranked.add(Map.entry(e.getKey(), e.getValue() / sum));
        ranked.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
        return ranked;
    }

    void seed() {
        String[][] data = {
            {"food", "Swiggy order biryani dinner"}, {"food", "Zomato lunch paid UPI"},
            {"food", "tea snacks canteen"}, {"food", "grocery vegetables kirana store"},
            {"food", "restaurant bill family dinner"},
            {"fuel", "HP petrol pump bike fuel"}, {"fuel", "Indian Oil petrol 500 debited"},
            {"fuel", "diesel filled car"}, {"fuel", "bike petrol refill"},
            {"client payment", "received from client project payment credited"},
            {"client payment", "invoice paid by client NEFT credited"},
            {"client payment", "advance received website project UPI credited"},
            {"client payment", "milestone payment received client"},
            {"tools", "hostinghat domain renewal debited"}, {"tools", "GitHub Copilot subscription"},
            {"tools", "VPS server DigitalOcean monthly bill"}, {"tools", "Google Play developer fee software license"},
            {"tools", "domain hosting SSL certificate purchase"},
            {"bills", "electricity bill paid"}, {"bills", "Jio recharge mobile plan"},
            {"bills", "broadband internet bill airtel"}, {"bills", "water bill and gas cylinder"},
            {"travel", "train ticket IRCTC booking"}, {"travel", "bus ticket Ranchi"},
            {"travel", "Ola Uber cab ride"}, {"travel", "flight ticket booking"},
        };
        for (String[] d : data) train(d[0], d[1]);
    }

    public static void main(String[] args) {
        ExpenseClassifier nb = new ExpenseClassifier();
        nb.seed();
        System.out.println("Expense categorizer ready. Categories: " + nb.docCounts.keySet());
        System.out.println("Type text to classify | '+ category: text' to teach | 'exit' to quit\n");

        Scanner in = new Scanner(System.in);
        while (true) {
            System.out.print("> ");
            if (!in.hasNextLine()) break;
            String line = in.nextLine().trim();
            if (line.equalsIgnoreCase("exit")) break;
            if (line.isEmpty()) continue;

            if (line.startsWith("+") && line.contains(":")) {
                int i = line.indexOf(':');
                nb.train(line.substring(1, i), line.substring(i + 1));
                System.out.println("  learned.");
                continue;
            }
            var ranked = nb.predict(line);
            var best = ranked.get(0);
            System.out.printf("  -> %s (%.0f%%)%n", best.getKey(), best.getValue() * 100);
            if (ranked.size() > 1)
                System.out.printf("     next: %s (%.0f%%)%n", ranked.get(1).getKey(), ranked.get(1).getValue() * 100);
        }
    }
}
