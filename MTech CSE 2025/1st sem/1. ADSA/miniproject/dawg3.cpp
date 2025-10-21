#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <functional>
#include <chrono>
#include <cctype>
#include <memory>
#include <cmath>
#include <climits>

using namespace std;
using steady_clock = chrono::steady_clock;
using millis = chrono::milliseconds;

// Configuration for user-frequency rotation
constexpr int MAX_USER_FREQ = 100000; // threshold

// ------------------------ Node ------------------------
struct Node {
    bool isFinal = false;
    int freq = 0;                 // original dictionary frequency
    int user_freq = 0;            // bounded user frequency (rotating)
    long long total_selects = 0;  // absolute count of selections
    int subtree_freq = 0;         // cumulative weight of subtree (uses combined weight)
    unordered_map<char, Node*> edges; // transitions
    int id = -1; // unique id
    Node(int _id = -1) : isFinal(false), freq(0), user_freq(0), total_selects(0), subtree_freq(0), id(_id) {}
};

// ------------------- MinimalDAWG class -------------------
class MinimalDAWG {
private:
    Node* root;
    long long nextId = 1;
    size_t node_count = 0;

    // register: signature -> canonical node pointer
    unordered_map<string, Node*> reg;

    vector<Node*> prev_path;
    string prev_word;

    Node* newNode() {
        Node* n = new Node(static_cast<int>(nextId++));
        ++node_count;
        return n;
    }

    // signature uses final flag and word freq (original) and children ids
    string nodeSignature(Node* n) {
        vector<pair<char,int>> items;
        items.reserve(n->edges.size());
        for (auto &kv : n->edges) {
            items.emplace_back(kv.first, kv.second->id);
        }
        sort(items.begin(), items.end());
        string s;
        s.push_back(n->isFinal ? '1' : '0');
        s.push_back('#');
        if (n->isFinal) {
            s += 'F';
            s += to_string(n->freq); // keep based on original dictionary freq (not user freq)
            s.push_back('#');
        }
        for (auto &it : items) {
            s.push_back(it.first);
            s.push_back(':');
            s += to_string(it.second);
            s.push_back(';');
        }
        return s;
    }

    void replace_or_register_last() {
        if (prev_path.size() <= 1) return;
        Node* node = prev_path.back();
        Node* parent = prev_path[prev_path.size() - 2];
        size_t D = prev_path.size() - 1;
        char incomingChar = prev_word[D - 1];
        string sig = nodeSignature(node);
        auto it = reg.find(sig);
        if (it != reg.end()) {
            Node* canon = it->second;
            parent->edges[incomingChar] = canon;
            delete node;
            --node_count;
        } else {
            reg.emplace(move(sig), node);
        }
        prev_path.pop_back();
    }

    void minimize_path_to(size_t downTo) {
        while (prev_path.size() > downTo) replace_or_register_last();
    }

    // Combined weight used for ranking: original freq + user_freq + long-term bias
    static int combined_weight(const Node* n) {
        if (!n) return 0;
        double bias = 0.0;
        if (n->total_selects > 0) {
            bias = log10((double)n->total_selects + 1.0) * 1000.0; // adjustable multiplier
        }
        double w = double(n->freq) + double(n->user_freq) + bias;
        if (w > (double)INT_MAX) return INT_MAX;
        return static_cast<int>(w);
    }

    // Compute subtree_freq for all canonical nodes (post-build or after updates)
    void compute_subtree_freq() {
        unordered_map<Node*, int> memo;
        function<int(Node*)> dfs = [&](Node* n) -> int {
            if (!n) return 0;
            auto it = memo.find(n);
            if (it != memo.end()) return it->second;
            int sum = n->isFinal ? combined_weight(n) : 0; // use combined weight for finals
            for (auto &kv : n->edges) {
                sum += dfs(kv.second);
            }
            memo[n] = sum;
            n->subtree_freq = sum;
            return sum;
        };
        dfs(root);
    }

public:
    MinimalDAWG() {
        root = newNode();
        prev_path.clear();
        prev_path.push_back(root);
        prev_word.clear();
    }

    ~MinimalDAWG() {
        unordered_set<Node*> seen;
        vector<Node*> stack;
        stack.push_back(root);
        while (!stack.empty()) {
            Node* n = stack.back(); stack.pop_back();
            if (!n || seen.count(n)) continue;
            seen.insert(n);
            for (auto &kv : n->edges) stack.push_back(kv.second);
            delete n;
        }
    }

    void insert_sorted(const string &word, int frequency = 1) {
        size_t common = 0;
        size_t minlen = min(prev_word.size(), word.size());
        while (common < minlen && prev_word[common] == word[common]) ++common;
        minimize_path_to(common + 1);
        Node* node = prev_path.back();
        for (size_t i = common; i < word.size(); ++i) {
            char c = word[i];
            Node* nxt = newNode();
            node->edges[c] = nxt;
            node = nxt;
            prev_path.push_back(node);
        }
        node->isFinal = true;
        node->freq = frequency; // dictionary-provided base frequency
        prev_word = word;
    }

    void finish_build() {
        minimize_path_to(1);
        // After minimization we compute cumulative subtree frequencies
        compute_subtree_freq();
    }

    // Hybrid Zipf + Length auto-weight helper:
    static int auto_weight_for_rank_and_word(size_t rank_zero_based, const string &word) {
        double base_double = 1000000.0 / double(rank_zero_based + 1); // 1e6 / (rank+1)
        int base = static_cast<int>(base_double);
        int len = static_cast<int>(word.size());
        int length_bonus = max(0, 100 - 5 * len);
        int weight = base + length_bonus;
        if (weight < 1) weight = 1;
        return weight;
    }

    // Load dictionary from file; supports either "word" or "word freq" per line.
    // If no freq present, assigns automatic weight (Hybrid Zipf+Length) based on rank after sorting.
    size_t load_from_file(const string &filename) {
        ifstream in(filename);
        if (!in.is_open()) return 0;

        vector<pair<string,int>> items;
        string line;
        size_t line_num = 0;

        // Read raw words (and optional frequencies)
        while (getline(in, line)) {
            ++line_num;
            size_t a = 0; while (a < line.size() && isspace((unsigned char)line[a])) ++a;
            size_t b = line.size(); while (b > a && isspace((unsigned char)line[b-1])) --b;
            if (b <= a) continue;
            string token = line.substr(a, b - a);
            stringstream ss(token);
            string w; ss >> w;
            int f = -1; // -1 means "not provided"
            if (ss >> f) {
                // user provided explicit freq
            } else {
                f = -1;
            }
            for (auto &ch : w) ch = tolower((unsigned char)ch);
            items.emplace_back(w, f);

            cout << "\rRead " << line_num << " lines..." << flush;
        }
        in.close();
        cout << "\rRead " << items.size() << " lines.           " << endl;

        if (items.empty()) return 0;

        // Sort words lexicographically (required by Daciuk algorithm)
        sort(items.begin(), items.end(), [](const pair<string,int>& A, const pair<string,int>& B){
            if (A.first != B.first) return A.first < B.first;
            return A.second > B.second;
        });

        // Assign automatic weights where needed using Hybrid Zipf+Length
        vector<pair<string,int>> final_items;
        final_items.reserve(items.size());
        for (size_t i = 0; i < items.size(); ++i) {
            const string &w = items[i].first;
            int f = items[i].second;
            if (f > 0) {
                // user-provided frequency preserved
            } else {
                f = auto_weight_for_rank_and_word(i, w);
            }
            final_items.emplace_back(w, f);
        }

        // Insert in sorted order
        for (auto &p : final_items) insert_sorted(p.first, p.second);
        finish_build();
        return final_items.size();
    }

    void load_sample() {
        vector<pair<string,int>> items = {
            {"action", 180},{"analysis", 70},{"banana", 250},{"band", 90},
            {"bank", 110},{"cat", 500},{"cater", 200},{"caterpillar", 50},
            {"concatenate", 40},{"concatenation", 60},{"dog", 700},{"dot", 150},
            {"dove", 80},{"nation", 120},{"station", 300}
        };
        sort(items.begin(), items.end());
        for (auto &p : items) insert_sorted(p.first, p.second);
        finish_build();
    }

    bool contains(const string &word) const {
        const Node* cur = root;
        for (char c : word) {
            auto it = cur->edges.find(c);
            if (it == cur->edges.end()) return false;
            cur = it->second;
        }
        return cur->isFinal;
    }

    // Return pointer to final node for a word (or nullptr if not found)
    Node* get_final_node(const string &word) const {
        Node* cur = root;
        for (char c : word) {
            auto it = cur->edges.find(c);
            if (it == cur->edges.end()) return nullptr;
            cur = it->second;
        }
        return cur->isFinal ? cur : nullptr;
    }

    // Frequency info: returns tuple (found, base_freq, user_freq, total_selects, combined)
    tuple<bool,int,int,long long,int> get_frequency_info(const string &word) const {
        const Node* cur = root;
        for (char c : word) {
            auto it = cur->edges.find(c);
            if (it == cur->edges.end()) return {false,0,0,0,0};
            cur = it->second;
        }
        if (!cur->isFinal) return {false,0,0,0,0};
        int combined = combined_weight(cur);
        return {true, cur->freq, cur->user_freq, cur->total_selects, combined};
    }

    // Increment user selection for a word (apply rotation rule) and recompute subtree weights
    bool select_word(const string &word) {
        Node* n = get_final_node(word);
        if (!n) return false;
        n->total_selects += 1;
        // increment user_freq with rotation preserving relative order
        long long new_user = (long long)n->user_freq + 1LL;
        if (new_user > MAX_USER_FREQ) {
            int half = MAX_USER_FREQ / 2; // e.g., 50k
            int wrapped = static_cast<int>(new_user % half);
            n->user_freq = half + wrapped;
        } else {
            n->user_freq = static_cast<int>(new_user);
        }
        // After changing user_freq/total_selects we need to recompute subtree weights
        compute_subtree_freq();
        return true;
    }

    int get_frequency(const string &word) const {
        const Node* cur = root;
        for (char c : word) {
            auto it = cur->edges.find(c);
            if (it == cur->edges.end()) return -1;
            cur = it->second;
        }
        return cur->isFinal ? combined_weight(cur) : -1;
    }

    // Weighted autocomplete: uses node->subtree_freq to guide search.
    // Returns up to k completions (word, combined_freq) sorted by combined frequency descending.
    vector<pair<string,int>> autocomplete_weighted(const string &prefix, size_t k = 10) const {
        vector<pair<string,int>> results;
        const Node* cur = root;
        for (char c : prefix) {
            auto it = cur->edges.find(c);
            if (it == cur->edges.end()) return results;
            cur = it->second;
        }

        struct Item {
            const Node* node;
            string suffix; // path from 'cur' node to this node
            int priority;  // node->subtree_freq
            bool operator<(const Item& other) const {
                if (priority != other.priority) return priority < other.priority;
                if (suffix.size() != other.suffix.size()) return suffix.size() > other.suffix.size();
                return suffix > other.suffix;
            }
        };

        priority_queue<Item> pq;
        pq.push({cur, string(), cur->subtree_freq});

        while (!pq.empty() && results.size() < k) {
            Item top = pq.top(); pq.pop();
            const Node* n = top.node;
            if (n->isFinal) {
                results.emplace_back(prefix + top.suffix, combined_weight(n));
            }
            for (auto &kv : n->edges) {
                const char ch = kv.first;
                const Node* child = kv.second;
                Item next{child, top.suffix + ch, child->subtree_freq};
                pq.push(move(next));
            }
        }

        sort(results.begin(), results.end(), [](const pair<string,int>& a, const pair<string,int>& b){
            if (a.second != b.second) return a.second > b.second;
            return a.first < b.first;
        });
        if (results.size() > k) results.resize(k);
        return results;
    }

    // Backwards-compatible name: public autocomplete uses weighted strategy
    vector<pair<string,int>> autocomplete(const string &prefix, size_t k = 10) const {
        return autocomplete_weighted(prefix, k);
    }

    size_t countNodes() const { return node_count; }

    size_t countEdges() const {
        unordered_set<const Node*> vis;
        vector<const Node*> st{root};
        size_t edges = 0;
        while (!st.empty()) {
            const Node* n = st.back(); st.pop_back();
            if (!n || vis.count(n)) continue;
            vis.insert(n);
            edges += n->edges.size();
            for (auto &kv : n->edges) st.push_back(kv.second);
        }
        return edges;
    }

    void print_structure(int maxNodes = 200) const {
        unordered_set<const Node*> vis;
        queue<const Node*> q;
        q.push(root);
        vis.insert(root);
        int cnt = 0;
        while (!q.empty() && cnt < maxNodes) {
            const Node* n = q.front(); q.pop();
            cout << "[" << n->id << "]";
            if (n->isFinal) cout << "(F:" << n->freq << ",U:" << n->user_freq << ",T:" << n->total_selects << ")";
            cout << "(S:" << n->subtree_freq << ")";
            cout << " -> ";
            for (auto &kv : n->edges) {
                cout << kv.first << "->" << kv.second->id << " ";
                if (!vis.count(kv.second)) { vis.insert(kv.second); q.push(kv.second); }
            }
            cout << "\n"; ++cnt;
        }
        if (!q.empty()) cout << "... (more nodes)\n";
    }

    // ---------------- Persistence for user frequencies ----------------
    // Saves lines: word user_freq total_selects
    bool save_user_freqs(const string &filename) const {
        ofstream out(filename);
        if (!out.is_open()) return false;
        // Collect all words and their user freq data
        vector<pair<string,const Node*>> words;
        string cur;
        function<void(const Node*)> dfs = [&](const Node* n) {
            if (!n) return;
            if (n->isFinal) words.emplace_back(cur, n);
            for (auto &kv : n->edges) {
                cur.push_back(kv.first);
                dfs(kv.second);
                cur.pop_back();
            }
        };
        dfs(root);
        for (auto &p : words) {
            const string &w = p.first;
            const Node* n = p.second;
            if (n->user_freq != 0 || n->total_selects != 0) {
                out << w << " " << n->user_freq << " " << n->total_selects << "\n";
            }
        }
        return true;
    }

    // Loads user frequencies (word user_freq total_selects)
    bool load_user_freqs(const string &filename) {
        ifstream in(filename);
        if (!in.is_open()) return false;
        string line;
        size_t cnt = 0;
        while (getline(in, line)) {
            stringstream ss(line);
            string w; long long uf = 0; long long ts = 0;
            if (!(ss >> w >> uf >> ts)) continue;
            Node* n = get_final_node(w);
            if (n) {
                n->user_freq = static_cast<int>(uf);
                n->total_selects = ts;
                ++cnt;
            }
        }
        in.close();
        if (cnt > 0) compute_subtree_freq();
        return true;
    }

    // ---------------- Spell-check helpers ----------------
    // Compute Levenshtein distance with cutoff (banded DP using two rows)
    // static int edit_distance_with_cutoff(const string &a, const string &b, int maxDist) {
    //     int na = (int)a.size(), nb = (int)b.size();
    //     if (abs(na - nb) > maxDist) return maxDist + 1;
    //     vector<int> prev(nb + 1), cur(nb + 1);
    //     for (int j = 0; j <= nb; ++j) prev[j] = j;
    //     for (int i = 1; i <= na; ++i) {
    //         cur[0] = i;
    //         int best = cur[0];
    //         int start = max(1, i - maxDist);
    //         int end = min(nb, i + maxDist);
    //         for (int j = start; j <= end; ++j) {
    //             int cost = (a[i-1] == b[j-1]) ? 0 : 1;
    //             int ins = cur[j-1] + 1;
    //             int del = prev[j] + 1;
    //             int rep = prev[j-1] + cost;
    //             cur[j] = min({ins, del, rep});
    //             best = min(best, cur[j]);
    //         }
    //         if (best > maxDist) return maxDist + 1;
    //         swap(prev, cur);
    //     }
    //     return prev[nb];
    // }

    // Collect all words and nodes (used by spell)
    // void collect_all_words(vector<pair<string,const Node*>> &out) const {
    //     string cur;
    //     function<void(const Node*)> dfs = [&](const Node* n) {
    //         if (!n) return;
    //         if (n->isFinal) out.emplace_back(cur, n);
    //         for (auto &kv : n->edges) {
    //             cur.push_back(kv.first);
    //             dfs(kv.second);
    //             cur.pop_back();
    //         }
    //     };
    //     dfs(root);
    // }

    // Spell-check: return up to k suggestions within maxDist edits, ordered by (distance, combined weight)
//     vector<pair<string,int>> spell_suggestions(const string &word, int maxDist = 2, size_t k = 10) const {
//         vector<pair<string,const Node*>> all;
//         collect_all_words(all);
//         vector<tuple<int,int,string>> cand; // (distance, -combined_weight, word) for sorting
//         for (auto &p : all) {
//             const string &w = p.first;
//             int d = edit_distance_with_cutoff(word, w, maxDist);
//             if (d <= maxDist) {
//                 int cw = combined_weight(p.second);
//                 cand.emplace_back(d, -cw, w);
//             }
//         }
//         sort(cand.begin(), cand.end());
//         vector<pair<string,int>> res;
//         for (size_t i = 0; i < cand.size() && res.size() < k; ++i) {
//             res.emplace_back(get<2>(cand[i]), -get<1>(cand[i]));
//         }
//         return res;
//     }
 };

// ------------------ CLI helpers ------------------
static void print_help() {
    cout << "Commands:\n";
    cout << "  load <filename>           - load dictionary file (word or 'word freq' per line)\n";
    cout << "  contains <word>           - exact lookup\n";
    cout << "  freq <word>               - get frequency (original, user, total, combined)\n";
    cout << "  autocomplete <prefix> [k] - show top-k completions by combined frequency (k default 10)\n";
    cout << "  select <word>             - mark a word as selected (increments user frequency)\n";
    // cout << "  spell <word> [k]          - spell-check suggestions (edit distance <= 2)\n";
    cout << "  stats                     - print node/edge counts\n";
    cout << "  structure                 - small structural dump (debug)\n";
    cout << "  save [filename]           - save user frequency data (default: user_freqs.txt)\n";
    cout << "  help                      - show this help\n";
    cout << "  exit                      - quit (saves user frequencies automatically)\n";
}

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);

    MinimalDAWG dawg;
    cout << "DAWG builder (Daciuk incremental) with adaptive user-frequency, spell-check, and rotation" << endl;
    cout << "Type 'help' for commands." << endl;

    const string defaultFile = "words.txt";
    const string userFreqFile = "user_freqs.txt";
    ifstream f(defaultFile);
    if (f) { f.close();
        cout << "Found '" << defaultFile << "'. Loading...\n" << flush;
        auto t0 = steady_clock::now();
        size_t n = dawg.load_from_file(defaultFile);
        auto t1 = steady_clock::now();
        cout << "Loaded " << n << " entries in " << chrono::duration_cast<millis>(t1 - t0).count() << " ms" << endl;
        // try load user frequencies if present
        if (dawg.load_user_freqs(userFreqFile)) {
            cout << "Loaded user frequencies from '" << userFreqFile << "'.\n";
        }
    } else {
        cout << "No '" << defaultFile << "' found. Loading small sample.\n" << flush;
        auto t0 = steady_clock::now();
        dawg.load_sample();
        auto t1 = steady_clock::now();
        cout << "Sample loaded in " << chrono::duration_cast<millis>(t1 - t0).count() << " ms" << endl;
    }

    string line;
    while (true) {
        cout << "> " << flush;
        if (!getline(cin, line)) break;
        if (line.empty()) continue;
        stringstream ss(line);
        string cmd; ss >> cmd;

        if (cmd == "exit" || cmd == "quit") {
            // save user freqs automatically
            if (dawg.save_user_freqs(userFreqFile)) cout << "User frequencies saved to '" << userFreqFile << "'.\n";
            cout << "Bye.\n";
            break;
        }
        else if (cmd == "help") print_help();
        else if (cmd == "load") {
            string fname; ss >> fname;
            if (fname.empty()) { cout << "Usage: load <filename>\n"; continue; }
            cout << "Loading '" << fname << "' ...\n" << flush;
            auto t0 = steady_clock::now();
            size_t n = dawg.load_from_file(fname);
            auto t1 = steady_clock::now();
            if (n == 0) cout << "No entries loaded (file missing or empty?)\n";
            else cout << "Loaded " << n << " entries in " << chrono::duration_cast<millis>(t1 - t0).count() << " ms\n";
            // attempt to load user freqs for this dictionary (if same path)
            if (dawg.load_user_freqs(userFreqFile)) cout << "Loaded user frequencies from '" << userFreqFile << "'.\n";
        }
        else if (cmd == "contains") {
            string w; ss >> w;
            for (char &c : w) c = tolower((unsigned char)c);
            cout << (dawg.contains(w) ? "YES\n" : "NO\n");
        }
        else if (cmd == "freq") {
            string w; ss >> w;
            for (char &c : w) c = tolower((unsigned char)c);
            auto info = dawg.get_frequency_info(w);
            bool found = get<0>(info);
            if (!found) { cout << "Not found\n"; continue; }
            int base = get<1>(info);
            int user = get<2>(info);
            long long total = get<3>(info);
            int combined = get<4>(info);
            cout << "Word: " << w << "\n";
            cout << "  Original frequency: " << base << "\n";
            cout << "  User frequency:     " << user << "\n";
            cout << "  Total selects:      " << total << "\n";
            cout << "  Combined weight:    " << combined << "\n";
        }
        else if (cmd == "autocomplete") {
            string pref; ss >> pref;
            for (char &c : pref) c = tolower((unsigned char)c);
            int k = 10; if (!(ss >> k)) k = 10;
            auto res = dawg.autocomplete(pref, max(1,k));
            if (res.empty()) cout << "(no completions)\n";
            else for (auto &p : res) cout << p.first << "    (" << p.second << ")\n";
        }
        else if (cmd == "select") {
            string w; ss >> w;
            for (char &c : w) c = tolower((unsigned char)c);
            if (dawg.select_word(w)) cout << "Selected '" << w << "' (frequency updated).\n";
            else cout << "Word not found (cannot select)\n";
        }
        // else if (cmd == "spell") {
        //     string w; ss >> w;
        //     for (char &c : w) c = tolower((unsigned char)c);
        //     int k = 10; if (!(ss >> k)) k = 10;
        //     auto sug = dawg.spell_suggestions(w, 2, max(1,k));
        //     if (sug.empty()) cout << "(no suggestions)\n";
        //     else for (auto &p : sug) cout << p.first << "    (" << p.second << ")\n";
        // }
        else if (cmd == "stats") {
            cout << "nodes: " << dawg.countNodes() << ", edges: " << dawg.countEdges() << "\n";
        }
        else if (cmd == "structure") {
            dawg.print_structure(200);
        }
        else if (cmd == "save") {
            string fname; ss >> fname;
            if (fname.empty()) fname = userFreqFile;
            if (dawg.save_user_freqs(fname)) cout << "Saved user frequencies to '" << fname << "'.\n";
            else cout << "Failed to save user frequencies to '" << fname << "'.\n";
        }
        else cout << "Unknown command. Type 'help' for commands.\n";
    }

    return 0;
}

