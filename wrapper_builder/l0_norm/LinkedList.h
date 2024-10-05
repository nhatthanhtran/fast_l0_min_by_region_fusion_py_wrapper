/*
Author:		Rang M. H. Nguyen
Reference:	Rang M. H. Nguyen, Michael S. Brown
			Fast and Effective L0 Gradient Minimization by Region Fusion
			ICCV 2015
Date:		Dec 1st, 2015
*/
//-------------------------------

#pragma once
#include <iostream>
struct Node2
{
	int value;
	Node2* next;
};

class LinkedList
{
public:
	Node2* pHead;
	Node2* pTail;
	void append(LinkedList&);
	void insert(int v);
	LinkedList(void);
	void clear();
	~LinkedList(void);
};

