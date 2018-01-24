#include <iostream>
#include <memory>
#include <vector>

template <typename T>
struct LnkLstNode1z {
    T val;
    std::shared_ptr<LnkLstNode1z<T>> next;

    LnkLstNode1z(): next(nullptr) {}
    LnkLstNode1z(T x): val(x), next(nullptr) {}
    LnkLstNode1z(std::vector<T> vec) {
        auto head = constr(vec);
        val = head->val;
        next = head->next;
    }
private:
    std::shared_ptr<LnkLstNode1z<T>> constr(std::vector<T> vec) {
        if (vec.empty()) return nullptr;
        auto ghost = std::make_shared<LnkLstNode1z<T>>();
        auto node = ghost;
        for (auto a: vec) {
            node = node->next = std::make_shared<LnkLstNode1z<T>>(a);
        }
        return ghost->next;
    }
};

template <typename T>
struct LnkLstNode {
    T val;
    LnkLstNode *next;
    LnkLstNode(T x): val(x), next(NULL) {}
    LnkLstNode(std::vector<T> arr) {
        LnkLstNode<T> *root = constr(arr);
        if (NULL != root) {
            val = root->val;
            next = root->next;
        } else {
            val = INT_MAX;
            next = NULL;
        }
    }
    ~LnkLstNode() { delTail(this); }
    void print() { print(this); }

private:
    LnkLstNode<T>* constr(std::vector<T> arr) {
        if (arr.empty()) return NULL;
        LnkLstNode<T> *root = new LnkLstNode<T>(arr[0]);
        LnkLstNode<T> *curr = root;
        for (auto it = arr.begin() + 1; it != arr.end(); ++it) {
            curr = curr->next = new LnkLstNode<T>(*it);
        }
        return root;
    }

    void delTail(LnkLstNode<T> *root) {
        if (NULL == root) return;
        delTail(root->next);
        root->next = NULL;
    }

    void print(LnkLstNode<T> *root) {
        if (NULL == root) return;
        std::cout << root->val;
        if (NULL == root->next)
            std::cout << " || " << std::endl;
        else
            std::cout << " -> ";
        print(root->next);
    }
};
