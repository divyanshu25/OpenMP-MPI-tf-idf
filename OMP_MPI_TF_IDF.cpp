#include<stdio.h>
#include<iostream>
#include<map>
#include<iostream>
#include<fstream>
#include<algorithm>
#include<mpi.h>
#include<dirent.h>
#include<string.h>
#include<string>
#include<cmath>
#include<stdlib.h>
#include<vector>
#include<omp.h>
#include <iomanip>
#define WMAX 1000
#define FMAX 3
#define pb push_back
#define mp make_pair
using namespace std;

//========================================================//
struct dirent* ent;
DIR* books;

typedef struct msg{
	int start,end;
}MSG;
//=========================================================//
bool myFunc(pair<string, float> a,pair<string, float>b){
	return a.second > b.second;
}
//========================================================//
bool sortFunc(pair<int,double>a, pair<int,double>b){
	return a.second < b.second;
}
//========================================================//
int main(int argc , char *argv[]){
	double startT, endT;
	const int MASTER  =  0;
	int numTasks,rank,count=0,datawait,rc;
	MPI_Status Stat;
	MPI_Request req;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numTasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);	
	omp_set_num_threads(4);
	/* create a custom datatype */
	const int nitems=2;
	map<string,float>IDF;
	map<string,float>::iterator it;
	int blocklengths[2] = {1,1};
	MPI_Datatype types[2] = {MPI_INT, MPI_INT};
	MPI_Datatype mpi_msg_type;
	MPI_Aint     offsets[2];

	offsets[0] = offsetof(MSG, start);
	offsets[1] = offsetof(MSG, end);

	MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_msg_type);
	MPI_Type_commit(&mpi_msg_type);
	
	/* MASTER CODE */
	MSG outMsg;
	MSG inMsg;
	if(rank == MASTER){
		startT = omp_get_wtime();
		if((books = opendir("BOOKS")) == NULL)
			printf("Directory Not open\n");
		while((ent = readdir(books))!= NULL){
			string str(ent->d_name);
			int pos = str.find(".txt");
			if(pos != string::npos){
				count++;
			}
		}
		if(numTasks > count){
			printf("More Number of processes then files in corpus\n");
			exit(0);
		}
		closedir(books);
		int range = (int)floor(count / (float)(count,numTasks-1));
		int s = 1,e;
		/* send out Messages to the slaves */
		for(int i = 1 ; i< numTasks ; i++){
			e = s + range - 1;
			outMsg.start = s;
			outMsg.end = e;
			if(i == numTasks - 1)
				outMsg.end = max(e,count);
			s = s+range;
			rc = MPI_Send(&outMsg, 1, mpi_msg_type, i, 1, MPI_COMM_WORLD);
	      		printf("Task %d: Sent message %d %dto task %d with tag 1\n",rank, outMsg.start, outMsg.end, i);
		}
		
		// wait for completion message from all idfs. // 
		for(int i = 1 ;  i< numTasks ; i++){
			rc = MPI_Recv(&inMsg, 1, mpi_msg_type, i, 1, MPI_COMM_WORLD, &Stat);
		}
		//printf("IDF re\n");
		
		// compute Global IDF //
		IDF.clear();
		char fname[80],temp[8],token[800];
		float val;
		int cnt;
		int flag = 0;
		#pragma omp parallel for private(fname,token,val,cnt)
		for(int i= 1 ; i< numTasks ; i++){
			sprintf(fname,"Output/IDF/idf%d.txt",i);
			FILE* fp = fopen(fname,"r");
			while(fscanf(fp,"%s%s%d",token,temp,&cnt) != EOF ){
				string str(token);
				if(IDF.find(str) == IDF.end()){
					#pragma omp critical
						IDF.insert(mp(str,cnt));
				}
				else{
					#pragma omp critical
					IDF[str] += val;
				}
			}	
		}
		ofstream idout;
		idout.open("Output/IDF/Corpus_Idf.txt");
		for(it = IDF.begin() ; it != IDF.end() ; it++){
			it -> second = log10((count) / it->second);
			idout << it -> first << " -> " << it -> second<<endl;
		}
		IDF.clear();
		for(int i = 1 ;  i< numTasks ; i++){
			rc = MPI_Send(&outMsg, 1, mpi_msg_type, i, 1, MPI_COMM_WORLD);
		}
		printf("sent\n");
		// wait for completion message from all processes. // 
		for(int i = 1 ;  i< numTasks ; i++){
			rc = MPI_Recv(&inMsg, 1, mpi_msg_type, i, 1, MPI_COMM_WORLD, &Stat);
		}
		map<string,vector<pair<int,float> > >Index;
		map<string,vector<pair<int,float> > >::iterator it;
		vector<pair<int,float> >::iterator it2;
		
		#pragma omp parallel for private(fname,val,token) shared(Index)
		for(int i = 1 ; i<= count ; i++){
			sprintf(fname,"Output/TopWords/topwords%d.txt",i);
			FILE* fp = fopen(fname,"r");
			if(fp == NULL){
				cout << "Directory not open "<< endl;
				exit(0);
			}
			while((fscanf(fp,"%s%s%f",token,temp,&val))!= EOF){
				string tok(token);
				if(Index.find(tok) == Index.end()){
					vector<pair<int , float> >te;
					te.pb(mp(i,val));
					#pragma omp critical
					Index.insert(mp(tok,te));
					
				}
				else{
					if(Index[tok].size() < FMAX)
						#pragma omp critical
						Index[tok].pb(mp(i,val));
				}
					
			}
			fclose(fp);
		}
		ofstream out;
		out.open("Output/Index/index.txt");
		for(it = Index.begin() ; it != Index.end() ; it++){
			out << setw(20);
			out << it -> first << " -> ";
			for(it2 = (it -> second).begin() ; it2 != (it->second).end() ; it2++){
				out << setw(5);
				out << it2->first<< " ";
			}
			out << endl;
		}		
		out.close();
		Index.clear();
		endT = omp_get_wtime();
		printf("Timetaken: %lf\n",endT-startT);	
	}
	else{
		double t1,t2;
		t1 = omp_get_wtime();
		map<string, int>TF;
		map<string, float>IDF;
		map<string, int>::iterator it;
		map<string, float>::iterator itF;
		IDF.clear();
		int cflag = 0;
		float val;
		string StopWords[400];
		char fname[80],token[800],outFile[80],temp[30];
		memset(fname,'\0',sizeof(fname));
		memset(outFile,'\0',sizeof(outFile));
		/* Read Stop Words From File */
		ifstream stop("StopWords.txt");
		int k = 0;
		while(getline(stop, StopWords[k])){
			k++;
		}
		
		
		rc = MPI_Recv(&inMsg, 1, mpi_msg_type, MASTER, 1, MPI_COMM_WORLD, &Stat);
		printf("Task %d: Received %d char(s) (%d %d) from task %d with tag %d \n",
		rank, count, inMsg.start, inMsg.end,Stat.MPI_SOURCE, Stat.MPI_TAG);
		int idfflag = 0;
		#pragma omp parallel for private(fname,outFile,TF,token,it,idfflag) shared(StopWords)
		for(int i = inMsg.start ; i<= inMsg.end; i++){
			idfflag = 0;
			memset(fname,'\0', sizeof(fname));
			memset(outFile,'\0', sizeof(outFile));
			TF.clear();
			
			sprintf(fname,"BOOKS/%d.txt",i);
			sprintf(outFile,"Output/TF/tf%d.txt",i);
			FILE* fp = fopen(fname,"r");
			ofstream out;
			out.open(outFile);
			if(fp == NULL){
				printf("Directory not open\n");
				exit(0);
			}
			int mergeFlag = 0;
			string final_token;
			char sp[2] = {' ','\0'};
			while((fscanf(fp,"%s",token))!= EOF){
				string tok(token);
				tok.erase(remove(tok.begin(), tok.end(), '#'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), ' '), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '>'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '|'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '='), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '+'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '_'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), ','), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '-'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '*'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), ';'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '"'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '.'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '!'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), ')'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '('), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), ']'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '['), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '&'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '$'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '?'), tok.end());
				//transform(tok.begin(), tok.end(), tok.begin(), ::tolower);
				
				int tokFlag= 0;
				string tt = tok;
				transform(tt.begin(), tt.end(), tt.begin(), ::tolower);
				for(int k = 0 ; k < 400 ; k++){
					if(tt.compare(StopWords[k]) == 0){
						tokFlag = 1;
						break;
					}
				}
				if(tokFlag == 1){
					continue;
				}
				
				if(tok[0]>= 65 && tok[0] <= 90 && mergeFlag == 0){
					final_token.clear();
					final_token = tok;
					mergeFlag = 1;
					continue;
				}
				else if(tok[0]>= 65 && tok[0] <= 90 && mergeFlag == 1 && final_token.size() < 100){
					final_token = final_token + "_" +tok;
					continue;
				}
				else if((tok[0]<65 || tok[0] >90) && mergeFlag == 1){
					tok.clear();
					tok = final_token;
					final_token.clear();
					mergeFlag = 0;
				}
				
				if((int)tok[0] == 39 || ((int)tok[0]>=48 && (int)tok[0]<=57)|| tok.size() == 0)
					continue;
				if(TF.find(tok) == TF.end()){
					TF.insert(pair<string, int>(tok,1));
				}
				else{
					TF[tok] +=1;
				}
			}
			for(it = TF.begin() ; it!= TF.end() ; it++){
				out << it->first << " -> "<< it->second << endl;
				if(IDF.find(it->first) == IDF.end()){
					#pragma omp critical
					IDF.insert(pair<string , float>(it->first, 1));
				}
				else{
					IDF[it ->first] +=1;
				} 
			}
			out.close();
			fclose(fp);
		}
		ofstream out_idf;
		sprintf(fname,"Output/IDF/idf%d.txt",rank);
		out_idf.open(fname);
		for(itF = IDF.begin() ; itF!= IDF.end() ; itF++){
			//itF -> second = log10((inMsg.end - inMsg.start + 1)/itF->second);
			out_idf << itF->first << " -> "<< itF->second << endl; 
		}
		IDF.clear();
		out_idf.close();
		MSG inM;
		rc = MPI_Send(&inM, 1, mpi_msg_type, MASTER, 1, MPI_COMM_WORLD);
		printf("Local Idf Completed Message sent to Master\n");
		rc = MPI_Recv(&inM, 1, mpi_msg_type, MASTER, 1, MPI_COMM_WORLD, &Stat);
		//cout << "RE" << endl;
		//cout << "Recieved IDF Generation Confirmation\n" << endl;
		FILE* idf_file;
		idf_file = fopen("Output/IDF/Corpus_Idf.txt","r");
		if(idf_file == NULL){
			cout << "Directory Not Open" << endl;
			exit(0);
		}
		while((fscanf(idf_file,"%s%s%f",token,fname,&val))!= EOF){
			string str2(token);
			IDF.insert(mp(str2, val));
		}
		vector<pair<string , float> >topWords;
		vector<pair<string , float> >::iterator itt;
		#pragma omp parallel for private(fname,itt,topWords,outFile)
		for(int i =  inMsg.start; i<= inMsg.end ; i++){
			int val;
			memset(fname,'\0',sizeof fname);
			sprintf(fname,"Output/TF/tf%d.txt",i);
			FILE* fp = fopen(fname,"r");
			if(fp == NULL){
				cout << "Directory not open\n";
			}
			while((fscanf(fp,"%s%s%d",token,temp,&val))!=EOF){
				string tok(token);
				
				//#pragma omp critical
				{
					if(IDF.find(tok) != IDF.end()){
						topWords.pb(mp(tok,val * IDF[tok]));
					}
				}
			}
			sort(topWords.begin(),topWords.end(),myFunc);
			memset(outFile,'\0',sizeof outFile);
			sprintf(outFile,"Output/TopWords/topwords%d.txt",i);
			ofstream out;
			out.open(outFile);
			int k = 0;
			for(itt = topWords.begin(); itt!= topWords.end() && k< WMAX ; itt++,k++){
				out << itt -> first << " -> " << itt -> second << endl;
			}
			out.close();
			topWords.clear();	
		}
		// completion Message
		t2 = omp_get_wtime();
		rc = MPI_Send(&outMsg, 1, mpi_msg_type, MASTER, 1, MPI_COMM_WORLD);
		printf("Time Taken by Process %d is %lf\n",rank,t2-t1);
	}
	MPI_Finalize();	
return 0;	
}
