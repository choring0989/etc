#include <stdio.h>
#include <stdlib.h>

typedef struct node {
	int key;
	struct node *next;
}queue;

int empty(queue*);//return 1 if empty
int enqueue(int, queue**, queue**);//return 1 if success
int dequeue(queue**);//return n if success
void print(queue*);//print all node

int main() {
	queue *pqueue = (queue*)malloc(sizeof(queue));
	queue *head = NULL;
	queue *tail = NULL;
	/*Can be arbitrarily changed*/
	enqueue(1, &head, &tail);
	enqueue(2, &head, &tail);
	enqueue(3, &head, &tail);
	enqueue(4, &head, &tail);
	enqueue(5, &head, &tail);
	enqueue(6, &head, &tail);
	enqueue(7, &head, &tail);
	print(tail);
	dequeue(&tail);
	dequeue(&tail);
	dequeue(&tail);
	dequeue(&tail);
	dequeue(&tail);
	dequeue(&tail);
	dequeue(&tail);
	dequeue(&tail);
	print(tail);

	return 1;
}

/*If there is no tail, it is empty.*/
int empty(queue *s) {
	if (s == NULL) return 1;
	else return 0;
}

int enqueue(int n, queue **h, queue **t) {
	queue *p;

	p = (queue*)malloc(sizeof(queue));

	p->key = n;
	p->next = NULL;

	if (*t == NULL)
	{
		*h = p;
		*t = p;
		return 1;
	}

	(*h)->next = p;
	*h = p;
	return 1;
}

int dequeue(queue **t) {
	queue *p = *t;
	int n;

	if (empty(*t)) {
		printf("Empty!\t");
		return 0;
	}

	n = p->key;

	printf("Dequeue! %d\n", n);

	*t = p->next;

	free(p);//delete node

	return n;
}

void print(queue *s) {

	if (empty(s)) {
		printf("NULL\n");
		exit(1);
	}

	queue *p = s;

	while (p != NULL)
	{
		printf("%d --> ", p->key);
		p = p->next;
	}
	printf("NULL\n");
}
