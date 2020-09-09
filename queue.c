#include <stdio.h>
#define MAX 10

int ds[MAX];
int front = 0;
int back = -1;

/* function prototypes for get, put, full, empty */
int dequeue(int *item);
int enqueue(int item);
int is_full();
int is_empty();

int main() {
	int n;
	enqueue(10);
	enqueue(40);
	enqueue(15);
	enqueue(35);
	dequeue(&n);
	dequeue(&n);
	enqueue(50);
	enqueue(20);
	enqueue(25);
	enqueue(50);
	dequeue(&n);
	dequeue(&n);
	dequeue(&n);
	dequeue(&n);
	dequeue(&n);
	dequeue(&n);
	return 1;
}
int dequeue(int *item) { /* get, may call empty */
	if (is_empty() == 1) return -1;
	*item = ds[front];
	printf("%d\n", *item);
	front++;
	return 0;
}
int enqueue(int newitem) { /* put, may call full */
	if (is_full() == 1) return -1;
	back++;
	ds[back] = newitem;
	return 0;
}
int is_full() {/*If back is 9, the queue is full.*/
	if (back >= (MAX - 1)) {
		printf("full!\n");
		return 1;
	}
	return 0;
}
int is_empty() {/*If back is -1, the queue is empty.*/
	if (back == -1) {
		printf("Empty!\n");
		return 1;
	}
	return 0;
}
