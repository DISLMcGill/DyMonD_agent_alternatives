#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/framework/ops.h>
#include <tensorflow/cc/framework/scope.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <unordered_set>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>


#include <boost/algorithm/string.hpp>
#include <map>
#include <pthread.h>
#include <mutex>
#include <queue>
#include <Utils.hpp>
#include <ctime>
#include <time.h>
#include <vector>
#include <sstream>
#include <pcap/pcap.h>
//#include "/usr/include/python3.5/Python.h" // "/home/melsaa1/anaconda3/envs/name/include/python3.7m/Python.h"
#include <agent.hpp> // needed for flow struct defn
#include <server.hpp> // needed for server method calls

//#define NUM_TESTS 5643
#define DIM1 100
#define DIM2 36
#define DIM3 1
#define OUT_DIM1 10
// DyMonD: "ten_classes_0/" and "/Users/izzychampion/tenclassesmodelpb0"
// "/Users/izzychampion/DyMonD/18classesmodelpb/model_0"
// ResNet: "/Users/izzychampion/DyMonD_Research/ResNet/resnetpb_/" or "/Users/izzychampion/resnetpb1_/"
// RNN: "/Users/izzychampion/DyMonD_Research/RNN/RNNpb"

#define MODEL_PATH "/Users/izzychampion/tenclassesmodelpb0"
#define SERVE_TAG "serve"
#define NUM_INPUTS 1 // number of inputs to model (not batch size)
#define NUM_OUTPUTS 1
// DyMond, Resnet: 4 // RNN: 3
#define NUM_DIMS 4
//#define BATCH_SIZE 5643
#define NUM_BATCHES 1
//#define N_DATA sizeof(float)*DIM1*DIM2*DIM3*BATCH_SIZE // number of bytes in one input
// DyMonD: "revamped.csv"
// ResNet: "/Users/izzychampion/DyMonD_Research/ResNet/ResnetTestData.csv"
// RNN: "/Users/izzychampion/DyMonD_Research/RNN/RNN_test_data.csv
#define PathToTestData "revamped.csv"
#define INPUT1_NAME "serving_default_conv2d_input"
// possibly >1 outputs 
#define OUTPUT1_NAME "StatefulPartitionedCall"

using namespace tensorflow;

using namespace std;
using namespace boost::filesystem;
struct service
{
  char ID[32];
  char label[32];
  float score;
  std::vector<std::string> URLS;
  std::string MSlabel;
  
};
float threshold = 0.98;
std::mutex mtx;
vector<struct flow*> flowarray;
queue<raw_pkt*> Que;
int enq=0;
int deq=0;
std::ofstream file;
char interface[32];
char* ipaddress = NULL;
char* tracefile = NULL; 
bool LiveMode=false;
double duration=30.0;
int FindService( vector < struct service *> &services, char ID [32])
{
 int pos=-1;
 for (int j=0; j< services.size();j++)
{ 
if(strcmp(services[j]->ID,ID)==0)
{
 
 pos=j;
 break;
}
}
return  pos;
}

vector<vector<string> > strTo2DStr(const string& str, const int& r, const int& c)
{
    vector<vector<string> > mat;
    int rows(r), cols(c);
    vector<string> words;
    istringstream ss(str);
    copy(istream_iterator<string>(ss),istream_iterator<string>(),back_inserter(words));

    int counter(0);
    for ( int i(0); i < rows; ++i ){
        vector<string> temp;
        for ( int j(0); j < cols; ++j){
            if ( counter < words.size() )
                temp.push_back(words[counter++]);
            else
                temp.push_back("");
        }
        mat.push_back(temp);
    }

    return mat;
}

void GetURLs(service* S, std::vector<char*> Packets)

{

    int methodCode;
    char *uri;
    size_t pos = 0;
    std::string token;
for (int i=0; i<Packets.size(); i++)
{

         methodCode = parseMethod(Packets[i], strlen(Packets[i]));

        if (methodsName[methodCode] != "NONE"){


   uri = parseUri(Packets[i], strlen(Packets[i]));
        if (uri !=NULL)
        {
        char * token = strtok(uri, "?");
        if (token!=NULL)
        {S->URLS.push_back((std::string)token); 
}
} 
}
}
}

std::string GetMSLabel(std::vector<std::string> URIS){
std::map<std::string, int> wordcount;
std::string word, data;
size_t pos = 0, pos1=0;
bool Threshold=false;
std::string MSlabel;
InitStopWords();
for (int i=0; i<URIS.size(); i++)
{
//count++;
boost::to_lower(URIS[i]);

std::string token, token1;
while ((pos = URIS[i].find( "/")) != std::string::npos) {
    token = URIS[i].substr(0, pos);
 if (token.find(".") != std::string::npos)  
{
std::vector<std::string> tokens;
while ((pos1 = token.find(".")) != std::string::npos) {
 token1 = token.substr(0, pos1);
    tokens.push_back(token1);
    token.erase(0, pos1 + 1);
}//while ((pos = data.find("."))
if (tokens.size()>0)
{
   if (SearchList(StopWords,tokens[tokens.size()-1]))
    {if (!tokens[tokens.size()-2].empty())word= tokens[tokens.size()-2];}
  else 
    word= tokens[tokens.size()-1];
 }  
}//if (token.find(".")
word=token;
if (!word.empty() && Alpha(word) && !(SearchList(StopWords,word)))
  {
    
if (wordcount.count(word)>0)
          wordcount[word] += 1;
else
  wordcount.insert ( std::pair<std::string,int>(word,1) );
  }
    URIS[i].erase(0, pos + 1);
}//while ((pos = data.find(delimiter))
}
for ( auto item : wordcount )
{
  if (((float)item.second/URIS.size()) >= 0.5)
      {MSlabel=MSlabel+"/"+item.first; Threshold=true;}
}
if(!Threshold)

{

    multimap<int, std::string> MM;
    for (auto& it : wordcount) {
        MM.insert(std::make_pair(it.second, it.first));
        //MM.insert({ it.second, it.first });
    }
map<int, std::string>::iterator itr;


  itr = MM.end();

  for (int i=0; i<3;i++)
{
  --itr;
  MSlabel=MSlabel+"/"+itr->second;
}
}

return MSlabel;
}

void *process_packet_queue(void*) {


    u_int8_t tcp_hl = 0;
    u_int16_t tcp_dl = 0;
    u_int8_t ip_hl=0;
    u_int16_t sport, dport;
    char buf[55];
    char* saddr = (char*)malloc(sizeof("aaa.bbb.ccc.ddd"));
    char* daddr = (char*)malloc(sizeof("aaa.bbb.ccc.ddd"));
    raw_pkt *rpkt = NULL;
    double run_duration;
    clock_t begin = clock();
if(LiveMode)
  run_duration=duration+0.1;//30.1;
else
  run_duration=duration;//100.0;
while (true) {

    clock_t end = clock();
        double elapsed_time = double(end - begin) / CLOCKS_PER_SEC;
        if (elapsed_time >= run_duration) {
            break;
        }

    if (deq < enq) {
        
        mtx.lock();
        rpkt = (raw_pkt *) Que.front();
        Que.pop();
        deq++;
        mtx.unlock();
 if (NULL != rpkt) {
            char *cp = rpkt->raw;

            ethhdr *eth_hdr = NULL;
            iphdr *ip_hdr = NULL;
            tcphdr *tcp_hdr = NULL;
            udphdr *udp_hdr = NULL;
            struct Ack_time *cap_time;

            int foundIndex = 0, tp, tp_l;
            bool Pfound = false;
            bool found = false;
            int b = 0;
            eth_hdr = packet_parse_ethhdr(cp);

            if (eth_hdr->ether_type != 0x0800) // not an IP packet
            {
                free_ethhdr(eth_hdr);
            }
            else {
                cp = cp + sizeof(ethhdr);
                ip_hdr = packet_parse_iphdr(cp);
                strncpy(saddr, ip_ntos(ip_hdr->saddr), sizeof("aaa.bbb.ccc.ddd"));
                strncpy(daddr, ip_ntos(ip_hdr->daddr), sizeof("aaa.bbb.ccc.ddd"));
                ip_hl = (ip_hdr->ihl) << 2; /* bytes */
                tp_l = ip_hdr->tot_len - ip_hl;
                tp = ip_hdr->protocol;
                cp = cp + ip_hl;
                if (tp == 17) // UDP packet
                {
                    udp_hdr = packet_parse_udphdr(cp);
                    sport = udp_hdr->uh_sport;
                    dport = udp_hdr->uh_dport;
                    tcp_dl = tp_l - 8;
                    cp = cp + 8;
                } else  // TCP packet
                {

                    tcp_hdr = packet_parse_tcphdr(cp);
                    sport = tcp_hdr->th_sport;
                    dport = tcp_hdr->th_dport;
                    tcp_hl = tcp_hdr->th_off << 2; /* bytes */
                    tcp_dl = tp_l - tcp_hl;
                    cp = cp + tcp_hl;
                }
                snprintf(buf, sizeof(buf), "%s-%s--%" PRIu16 "-%" PRIu16 "-%d", saddr, daddr, sport, dport, tp);
                for (int i = 0; i < flowarray.size(); i++) {
                    if (flowarray[i]->flowID == NULL) continue;
                    if (strstr(buf, flowarray[i]->flowID) != NULL) {
                        found = true;
                        flowarray[i]->NumBytes += tcp_dl;
                        if (strstr(flowarray[i]->proto, "Unknown") != NULL) {
                            Pfound = true;
                        }

                        foundIndex = i;
                        break;
                    }
                }

                if (!found) {
                    // allocate memory for one `struct flow'
                    struct flow *f = (struct flow *) calloc(sizeof(struct flow), 1);
                    // copy the data into the new element (structure)
                    f->flowID = strdup(buf);
                    f->saddr = strdup(saddr);
                    f->daddr = strdup(daddr);
                    f->sport = new char[sizeof(sport) + 1];
                    f->dport = new char[sizeof(dport) + 1];
                    sprintf(f->sport, "%u", sport);
                    sprintf(f->dport, "%u", dport);
                    f->NumBytes = tcp_dl;
                    f->protof = false;
                    strncpy(f->proto, "Unknown", 32);
                    flowarray.push_back(f);
                    Pfound = false;
                    foundIndex = flowarray.size() - 1;

                }
                if (tcp_hdr != NULL) {
                    if (tcp_hdr->th_flags == TH_ACK ||tcp_hdr->th_flags == 0x18) {
                        cap_time = (struct Ack_time *) calloc(sizeof(struct Ack_time), 1);
                        cap_time->sec = rpkt->pkthdr.ts.tv_sec;
                        cap_time->usec = rpkt->pkthdr.ts.tv_usec;
                        flowarray[foundIndex]->Ack_times.push_back(cap_time);
                    }
                }
               

if (flowarray[foundIndex]->Packets.size() < 100 && tcp_dl > 0)
{
char *linesp, *lineep, *dataend, *eol;
int lnl=0;
linesp = (char*)  cp;
dataend= cp +tcp_dl;
        lineep = find_line_end(linesp, dataend, (const char**)&eol);
        if(lineep!=NULL){
        lnl = lineep - linesp + 1;
        char *array = new char[lnl];
        strncpy(array,cp,lnl);
flowarray[foundIndex]->Packets.push_back(array);

}
else
{
char *array = new char[36];

if ( tcp_dl >= 36&& flowarray[foundIndex]->Packets.size() < 100)
 {        
                        strncpy(array, cp, 36);
                        flowarray[foundIndex]->Packets.push_back(array);
 } 
else if ( tcp_dl > 0&& tcp_dl<36 && flowarray[foundIndex]->Packets.size() < 100 )
            {
             int d=36-tcp_dl;int index=tcp_dl;
             strncpy(array,cp,index);
             for (int j=0; j<d; j++)
              {array[index]='0'; index++;}

flowarray[foundIndex]->Packets.push_back(array);
}
}
}



                if (tcp_hdr != NULL)
                    free_tcphdr(tcp_hdr);
                if (udp_hdr != NULL)
                    free_udphdr(udp_hdr);
                free_ethhdr(eth_hdr);
                free_iphdr(ip_hdr);


            } 

            raw_packet_free(rpkt);

        } else
           continue;

    } 
}
    
    free(saddr);
    free(daddr);
return 0;
}

void * 
capture_main(void*) {

    char errbuf[PCAP_ERRBUF_SIZE];
    memset(errbuf, 0, PCAP_ERRBUF_SIZE);
    struct pcap_pkthdr pkthdr;
    char *raw = NULL;
    pcap_t *cap = NULL;
    raw_pkt *pkt = NULL;
    clock_t begin, end;
    double elapsed_time;
    double sniff_duration = duration;//30.0;
    // added
    LiveMode = false;
    if (!LiveMode)
    {    printf("Processing the network trace file...\n");cap = pcap_open_offline(tracefile, errbuf);}
    else
    { 
       char *interface_pcap = interface;
       printf("interface %s\n" ,interface);
       printf("Starting to sniff....\n"); cap = pcap_open_live(interface_pcap, 65535, 1, 1000, errbuf);
       }  
        
         if (cap == NULL) {
        printf("errbuf: ");
        printf("%s\n", errbuf);
        exit(1);
    }
    begin = clock();
    raw = (char *) pcap_next(cap, &(pkthdr));
    while (NULL != raw) {
        pkt = MALLOC(raw_pkt, 1);
        pkt->pkthdr = pkthdr;
        char *r = MALLOC(char, pkthdr.len);
        memcpy(r, raw, pkthdr.len);
        pkt->raw = r;

        mtx.lock();
        Que.push(pkt);
        enq++;
        mtx.unlock();

       if (LiveMode)
{ 
        end = clock();
        elapsed_time = double(end - begin) / CLOCKS_PER_SEC;
        if (elapsed_time >= sniff_duration) {
            break;
        }
}

        raw = (char *) pcap_next(cap, &(pkthdr));
    }

    if (cap != NULL)
        pcap_close(cap);
return 0;
}

/***********************************
 
 ***********************************
 
 **** MAIN STARTS HERE ****
 
 ***********************************
***********************************/
int main(int argc, char *argv[]){
    std::ofstream myfile, FP, file;
    std::ifstream infile("services.txt");
    int b = 0;  int opt;
    void *thread_result;
    pthread_t job_pkt_q;
    pthread_t capture;
    char ID[28];
    string log_str = "DyMonD/logs/logging.txt"; // for debugging
    char *input_ip;
    /*
    wchar_t** _argv = (wchar_t**)PyMem_Malloc(sizeof(wchar_t*)*argc);
    for (int i=0; i<argc; i++) {
    wchar_t* argp = Py_DecodeLocale(argv[i], NULL); // already a variable called arg elsewhere
    _argv[i] = argp;
    }
    clock_t start1 = clock();
    Py_Initialize();
    PyObject * pModule = NULL;
    PyObject * pFunc = NULL;
    PyObject *pDict = NULL;
    PyObject *pReturn = NULL;
   Py_SetProgramName(_argv[0]); 
  PySys_SetArgv(argc, _argv); // must call this to get sys.argv and relative imports
    pModule = PyImport_ImportModule("Model");
        if(pModule==NULL){
                printf("Model is not found\n");
                PyErr_Print();}
                pDict = PyModule_GetDict(pModule); 
    pFunc = PyDict_GetItemString(pDict, "prediction");
    if(!pFunc ||!(PyCallable_Check(pFunc))){
        if (PyErr_Occurred())
                            PyErr_Print();
                        fprintf(stderr, "Cannot find prediction function \"%s\"\n", argv[2]);
        Py_XDECREF(pFunc);
                Py_DECREF(pModule);
        return 0;
    } 
    PyObject *PyList  = PyList_New(0);
    PyObject *ArgList = PyTuple_New(1);
*/
    //clock_t end1 = clock();
    //double elapsed1 = double(end1 - start1)/CLOCKS_PER_SEC;
 

InitMethodName();
// Standalone args and sniffing time
bool standalone = true;
while((opt = getopt(argc, argv, "t:i:f:p")) != -1){
                switch(opt){
            case 't':
                if(atof(optarg) <= 5 || atof(optarg) >= 1000) {
                    printf("Time out of range");
                    exit(EXIT_FAILURE);
                } else duration = atof(optarg);
                break;
            case 'i':
                    input_ip = optarg; break;
            case 'p':
                    ipaddress = optarg; break;
            case 'f':
                    tracefile = optarg; break;
                }
        }

char mode_buf[64], log[64], arg[64], time[64];
bool sniff_more = true;
string capture_dir; // = "captures/"
map<string, string> ip_map;
// added:

if(argc == 1 || strstr(argv[1], "-t") != NULL){
    standalone = false;
    printf("hello %d\n", argc);
    setup_server(); // prepare server for incoming client tcp connection
} else { // Standalone mode
    log[0] = '\0';
}
    ifstream inFile("interfaces/Interfaces.csv", ios::in);
    string lineStr;
    while (getline(inFile, lineStr))
    {
        // Interface is VALUE, IP is KEY
        int index = lineStr.find(" ");
        string interface_val = lineStr.substr(0, index);
        string ip_address_key = lineStr.substr(index+1, lineStr.size()-1);
        ip_map[ip_address_key] = interface_val;
    }


    // This is for logging the flows sent over tcp for debugging purposes
    FP.open(log_str, std::ios_base::out);
    FP.close();
    
    while(sniff_more){
        if(standalone){
            printf("Running as standalone\n");
            sniff_more = false;
            if (input_ip != NULL) {
                LiveMode=true;
                strcpy(arg, input_ip);
                strncpy(interface, ip_map[arg].c_str(), 32);
            }
        } else {
            receive_message(mode_buf, false); // receive indication if using interface or reading from file
            if(!strcmp(mode_buf, "stop")) {
                break;
            }
            receive_message(log, true); // receive indication if sending via tcp or writing to logfile
            receive_message(arg, true);  // receive ip address or name of pcap file
            receive_message(time, true);
            duration = atof(time);
            if(mode_buf[0] == 'i'){
                printf("Monitoring request received\n");
                LiveMode=true;
                strncpy(interface, ip_map[arg].c_str(), 32);
                flowarray.clear();
            } else {
                LiveMode=false;
                capture_dir.clear();
                capture_dir.append("captures/").append(arg);
                tracefile = (char*)capture_dir.c_str();
                sniff_more = false;
            }
        }
        

        /* Start packet receiving thread */
        pthread_create(&job_pkt_q, NULL,&process_packet_queue, NULL);
        /* Start main capture in live or offline mode */
        pthread_create(&capture, NULL, &capture_main, NULL);

        // Wait for all threads to finish
        pthread_join(job_pkt_q, &thread_result);
    pthread_join(capture, &thread_result);


    // TENSORFLOW C++ API
    //*** INITIAL PROCESSING ***//
    auto total_process = std::chrono::steady_clock::now();
    auto init_processing = std::chrono::steady_clock::now();	
	
    const std::string export_dir = MODEL_PATH;	
	tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;
	tensorflow::SavedModelBundle model_bundle;

    //f_session.reset(NewSession(options));
    model_bundle.session.reset(NewSession(session_options));

    auto init_processing_end = std::chrono::steady_clock::now();	
    auto init_proc = std::chrono::duration_cast<std::chrono::nanoseconds>(init_processing_end - init_processing).count();
    std::cout << "Initial processing: " << init_proc << "nanoseconds" << std::endl;

    auto session_start = std::chrono::steady_clock::now();	

	tensorflow::Status status = tensorflow::LoadSavedModel(
        	session_options, run_options, export_dir, {tensorflow::kSavedModelTagServe},
        	&model_bundle);

	auto session_end = std::chrono::steady_clock::now();	
    auto cr_session = std::chrono::duration_cast<std::chrono::nanoseconds>(session_end - session_start).count();
    std::cout << "Create a session: " << cr_session << "nanoseconds" << std::endl;
   
    //*** CHECK IF ERROR CREATING SESSION ***//

    if (!status.ok()) {
    		std::cout << status.ToString() << "\n";
    		return 1;
  	} else {
    	std::cout << "Load saved model successfully" << std::endl;
  	}
    
    //*** GET DATA ***//

    clock_t start2 = clock();

    char *array = new char[36];
    int rownum = 0;
    for (int i = 0; i < flowarray.size(); i++) {
           if (flowarray[i]->Packets.size() == 100 ) {
               rownum++;
           }
    }
    int NUM_TESTS = rownum;
    int* array_tens = (int*)malloc(rownum*DIM1*DIM2*DIM3*sizeof(int));
    //int* array_tens = static_cast<int*>(malloc(rownum*DIM1*DIM2*DIM3 * sizeof(int)));
    int *arr_2d=new int[rownum*3600];
    //int(*p)[3600]=(int(*)[3600])arr_2d; 

    int itr_row = 0;
    int itr_col = 0;
    // ADDED:
    int h = 0;
    for (int i = 0; i < flowarray.size(); i++) {

        if (flowarray[i]->Packets.size() == 100 ) {
             for (int j = 0; j < 100; j++) {     
                       strncpy(array, flowarray[i]->Packets[j], 36);
                for (int l = 0; l < 36; l++) {   
                        
                        int b = (unsigned char)array[l]; 
                        //p[itr_row][itr_col+l]= b;
                        
                        array_tens[h] = b;
                        //printf("printing b %d\n", array_tens[h]);
                        h++;
                       
                    }
                    itr_col = itr_col + 36;

                }
                itr_row++;
                itr_col=0;
        }
    }
      


    clock_t end2 = clock();
    double elapsed2 = double(end2 - start2)/CLOCKS_PER_SEC; // Time needed to formulate the inpur to the deep learning model
    
    //*** MIN MAX NORMALIZATION ***//
    auto min_max_start = std::chrono::steady_clock::now();	
    int* max_vals = (int*)malloc(3600*sizeof(int));
    
    std::memset(max_vals, 0, sizeof(int)*3600);
   
    float* min_vals = (float*)malloc(3600*sizeof(float));
    std::memset(min_vals, 0, sizeof(int)*3600);
    float min_val;
    float max_val;
  
    for (int i = 0 ; i < NUM_TESTS*DIM1*DIM2*DIM3; i ++) {
        //if(array_tens[i] > max_val) max_val = array_tens[i];
        //if(array_tens[i] < min_val) min_val = array_tens[i];
        
        if(array_tens[i] > max_vals[i%3600]) max_vals[i%3600] = array_tens[i];
        if(array_tens[i] < min_vals[i%3600]) min_vals[i%3600] = array_tens[i];
        if(max_vals[i%3600] == 0 )max_vals[i%3600] = 1;
    } 
    
    // normalize
    for(int i = 0 ; i < NUM_TESTS*DIM1*DIM2*DIM3; i ++){
        //array_tens[i] = (array_tens[i]-min_val)/(max_val - min_val);
        if(max_vals[i%3600] != 0 && array_tens[i] != 0){
            array_tens[i] = (array_tens[i]-min_vals[i%3600])/(max_vals[i%3600] - min_vals[i%3600]);
        }
    }

    free(max_vals);
    auto min_max_end = std::chrono::steady_clock::now();
    auto normalizing = std::chrono::duration_cast<std::chrono::nanoseconds>(min_max_end - min_max_start).count();
    std::cout << "Min Max Normalization: " << normalizing << "nanoseconds" << std::endl;    

    int BATCH_SIZE = rownum;
    
    //*** GET INPUT & OUTPUT TENSORS ***//
    auto make_input = std::chrono::steady_clock::now();	
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT,tensorflow::TensorShape({BATCH_SIZE,DIM1,DIM2,DIM3}) );
	
    auto input_tensor_mapped  = input_tensor.tensor<float, NUM_DIMS>(); // to hold input data
    std::vector<tensorflow::Tensor> outputs;
   
    auto make_input_end = std::chrono::steady_clock::now();	
    auto make_input_tensor = std::chrono::duration_cast<std::chrono::nanoseconds>(make_input_end - make_input).count();
    std::cout << "Make input/output tensors: " << make_input_tensor << "nanoseconds" << std::endl;


    std::string result_tens;
    long formatting_times = 0;
    long session_run_times = 0;

    std::ofstream output_file("output_of_agent.txt", std::ios::app);

    for(int i=0; i < NUM_BATCHES ; i++){
        
        //*** SHAPING INPUT DATA ***//
		auto format_data = std::chrono::steady_clock::now();
    	h = 0;
    	for(int j = 0; j < BATCH_SIZE; j++ ){
           
    		for (int m = 0 ; m < DIM1 ; m++){
				for(int n = 0 ; n < DIM2 ; n++ ){
                   
					for(int o = 0; o < DIM3 ; o++ ){
                        
                        input_tensor_mapped(j,m,n,o) = array_tens[j*DIM1*DIM2*DIM3 + m*DIM2*DIM3 + n*DIM3 + o];
                        
                		h++;
					} 

				}
			}
    	}

        auto format_data_end = std::chrono::steady_clock::now();
		auto data_format_time = std::chrono::duration_cast<std::chrono::nanoseconds>(format_data_end - format_data).count();
        std::cout << "Format Data: " << data_format_time << "nanoseconds" << std::endl;
		//*** RUN SESSION ***//
		auto start_run = std::chrono::steady_clock::now();	
	
		status = model_bundle.session->Run({{"serving_default_conv2d_input:0",input_tensor}}, {"StatefulPartitionedCall:0"}, {}, &outputs);
  		
        auto end_run = std::chrono::steady_clock::now();
        auto session_run = std::chrono::duration_cast<std::chrono::nanoseconds>(end_run - start_run).count();
        std::cout << "Session run: " << session_run << "nanoseconds" << std::endl;


        //session_run_times += static_cast<float>(session_run);

        //*** CHECK IF SESSION OK ***//
		if (!status.ok()) {
    		std::cout << status.ToString() << "\n";
    		return 1;
        }

        //*** WRITE OUTPUT TO FILE***//
		tensorflow::Tensor& output_tensor = outputs[0];
        auto output_mapped = output_tensor.tensor<float, 2>();
		
        // loop to get maximum
		for( int x = 0; x < BATCH_SIZE ; x++){
            float max = -1.0;
            int max_dim = 0;
			for(int  y = 0 ; y < OUT_DIM1 ; y ++ ){
                output_file << output_mapped(x,y);
                if (y < OUT_DIM1-1){
                    output_file << " ";
                }
                if (output_mapped(x,y) > max) {
                     max = output_mapped(x,y);
                     max_dim = y;
                }
			} output_file << "\n";
            result_tens += " ";
            result_tens += std::to_string(max_dim);
            result_tens += " ";
            result_tens += std::to_string(max);
            
		} 

	}
    outputs.clear();
    
    free(array_tens);
    tensorflow::Status close_status = model_bundle.session->Close();	
    output_file.close();
    if (!close_status.ok()) {
    		std::cerr << "Error closing session: " << close_status.ToString() << "\n";
	}
    
    auto end_total = std::chrono::steady_clock::now();	
    auto the_total = std::chrono::duration_cast<std::chrono::nanoseconds>(end_total - total_process).count();
    std::cout << "The whole process: " << the_total << "nanoseconds" << std::endl;


    /*
    std::string str;
   for (int x =0;x<rownum;x++) {
       for(int y = 0;y<3600;y++){
        str += std::to_string(p[x][y]);
        str += " ";
        }
   }*/

   // To view string being sent to model for this round of flows, save str to file here
    /*
    printf("Sending flows data to the model, starting service identification.\n");
    clock_t start3 = clock();
    PyTuple_SetItem(ArgList, 0, Py_BuildValue("s", str.c_str()));
    pReturn=PyObject_CallObject(pFunc, ArgList);
    clock_t end3 = clock();
    double elapsed3 = double(end3 - start3)/CLOCKS_PER_SEC;// Service Identification time
    
    PyArg_Parse(pReturn,"s",&result);
    */
    const char* result = result_tens.c_str();
    printf("result %s\n", result);
    int cols=2;
    vector< vector<string> > mat = strTo2DStr(result,rownum,cols);  
    //const char *label[18] = {"Cass-C", "Cass-S", "CassMN", "DB2-C", "DB2-S", "HTTP-S", "HTTP-C", "MYSQL-S", "MYSQL-C", "Memcached-C", "Memcached-S", "MonetDB-C", "MonetDB-S", "PGSQL-C", "PGSQL-S", "Redis-C", "Redis-S", "Spark-W"};
    
    const char *label[10] = {"Cass", "CassMN", "DB2", "HTTP", "Memcached", "MonetDB", "MySql", "PostgreSQL", "Redis", "Spark-W"};

    int counter_mat = 0;
    myfile.open("predictions.txt", std::ios_base::app);
    myfile << interface << "\n";

    for (int i = 0; i < flowarray.size(); i++) {
        
        if (flowarray[i]->Packets.size() == 100 ) {
            int index = stoi(mat[counter_mat][0]);
            printf("index %d\n", index);
            const char* lab=label[index];
            
            double score_double = std::stod(mat[counter_mat][1]);
            flowarray[i]->score=score_double;
            myfile << flowarray[i]->saddr << ":" << flowarray[i]->sport << " " << flowarray[i]->daddr << ":"
                    << flowarray[i]->dport <<" " << lab << " "<<  flowarray[i]->score << "\n";
            counter_mat++;
        }
    } 
myfile.close();
/******************validate label**********************/
vector < struct service *>services;

int counter_f = 0;
for(int i = 0; i < flowarray.size(); i++)
  {
     
      if(flowarray[i]->Packets.size() == 100 )
      {
        int ind = std::stoi(mat[counter_f][0]);
        string lab =label[ind];
        string lab_del = lab.substr(0, lab.size()-2);
        char * mat_lab = const_cast<char*>(lab_del.c_str());
        float mat_score = std::stod(mat[counter_f][1]);
        if(mat_score>=threshold){
            char *ip;
            char *port;
            int specialType=0;
            if(lab.back()=='S')
            {
              ip = flowarray[i]->saddr;
              port = flowarray[i]->sport;
              flowarray[i]->isServer=1;
        }
            else if(lab.back()=='C')
            {
              ip = flowarray[i]->daddr;
              port = flowarray[i]->dport;
              flowarray[i]->isServer=0;
            }
            else if(lab.back()=='N'||lab.back()=='W')
            {
              ip = flowarray[i]->daddr;
              port = flowarray[i]->dport;
              flowarray[i]->isServer=0;
              specialType=1;
            }
            char ID [32];
            strncpy (ID, ip,32);
            strncat (ID, port,32);
        
            int found = 0;
            int pos = 0;
            //check if server IP/port number is in services;
            for (int j = 0; j < services.size (); j++)
            {
              if (strcmp(services[j]->ID,ID)==0)
                {
                  found = 1;
                  pos = j;
                  break;
                }
            }
          if (found == 0 && mat_score >= threshold)
            {
              struct service *ser =(struct service *) calloc (sizeof (struct service), 1);
              strncpy(ser->ID,ID,32);
              if(specialType==0){
              strncpy(ser->label,mat_lab,32);}
              else{
                  strncpy(ser->label,const_cast<char*>(lab.c_str()),32);}
              ser->score = mat_score;
              services.push_back (ser);
            }
          else if(found==1){
          if (mat_score > services[pos]->score) { 
            if(specialType==0){strncpy(services[pos]->label,mat_lab,32);}
            else{strncpy(services[pos]->label,const_cast<char*>(lab.c_str()),32);}
          services[pos]->score = mat_score;
            }
        }
        }
        else{flowarray[i]->isServer=2;}
      
    counter_f++;
        }
    
    
}
/*for all flows
1 packets.size!=100 --> label unknown
2 packets.size =100 -->
                    1) found services.ID=flows.ID, update label if not equal
                    2) not found service.ID = flows.ID, label as unknown.
*/

/**************/


int count = 0;
for (int i = 0; i < flowarray.size(); i++)
    {
      if (flowarray[i]->Packets.size() != 100 )  {          
          strncpy(flowarray[i]->proto,"Unknown",32);   
      }

      else if (flowarray[i]->Packets.size() == 100 )
        {
        
        int ind = std::stoi(mat[count][0]);
        string flab =label[ind];
        string flab_del = flab.substr(0, flab.size()-2);
        char * mat_lab = const_cast<char*>(flab_del.c_str());
            char *ip;
            char *port;
            char *ip_1;
            char *port_1;
            ip = flowarray[i]->saddr;
            port = flowarray[i]->sport;
            ip_1 = flowarray[i]->daddr;
            port_1 = flowarray[i]->dport;
            char ID[32];
            strncpy (ID, ip,32);
            strncat (ID, port,32);
            char ID_1[32];
            strncpy (ID_1, ip_1,32);
            strncat (ID_1, port_1,32);
            int pos=-1;
            int found=0;
          for (int j = 0; j < services.size(); j++)
            {
              if (strcmp(services[j]->ID,ID)==0)
                {
                  pos = j;
                  found = 1;
                  break;
                }
            }
          int pos_1=-1;
          int found_1=0;
          for (int j = 0; j < services.size(); j++)
            {
              if (strcmp(services[j]->ID,ID_1)==0)
                {
                  pos_1 = j;
                  found_1 = 1;
                  break;
                }
            }
            if(found==1&&found_1==0){
                flowarray[i]->isServer = 1;
                if(strcmp(flowarray[i]->proto,services[pos]->label)!=0){
                    strncpy(flowarray[i]->proto,services[pos]->label,32);
                    flowarray[i]->score=services[pos]->score;
                    if(strcmp(flowarray[i]->proto,"CassMN")==0||strcmp(flowarray[i]->proto,"Spark-W")==0){
                        flowarray[i]->specialType=2;
                    }
                    else{flowarray[i]->specialType=1;}
                    }
                
            }
            else if(found==0&&found_1==1){
                flowarray[i]->isServer = 0;
                if(strcmp(flowarray[i]->proto,services[pos_1]->label)!=0){
                    strncpy(flowarray[i]->proto,services[pos_1]->label,32);
                     flowarray[i]->score=services[pos_1]->score;
                    if(strcmp(flowarray[i]->proto,"CassMN")==0||strcmp(flowarray[i]->proto,"Spark-W")==0){
                        flowarray[i]->specialType=2;
                    }
                    else{flowarray[i]->specialType=1;}

                    }

            }
            else if(found==1&&found_1==1){
                if(services[pos]->score > services[pos_1]->score){
                    flowarray[i]->isServer = 1;
                    if(strcmp(flowarray[i]->proto,services[pos]->label)!=0){
                    strncpy(flowarray[i]->proto,services[pos]->label,32);
                    flowarray[i]->score=services[pos]->score;
                    if(strcmp(flowarray[i]->proto,"CassMN")==0||strcmp(flowarray[i]->proto,"Spark-W")==0){
                        flowarray[i]->specialType=2;
                    }
                    else{flowarray[i]->specialType=1;}
                    }
                }
                else if(services[pos]->score <= services[pos_1]->score){
                     flowarray[i]->isServer = 0;
                    if(strcmp(flowarray[i]->proto,services[pos_1]->label)!=0){
                    strncpy(flowarray[i]->proto,services[pos_1]->label,32);
                    flowarray[i]->score=services[pos_1]->score;
                    if(strcmp(flowarray[i]->proto,"CassMN")==0||strcmp(flowarray[i]->proto,"Spark-W")==0){
                        flowarray[i]->specialType=2;
                    }
                    else{flowarray[i]->specialType=1;}
                    }
                }

            }
            else{
                flowarray[i]->specialType=3;
            }

       count ++;
        }


        
    }
// validate the Bidirection flows 

char RFlowID[55];
int test_counter=0;
for(int i = 0; i < flowarray.size (); i++){
    if(flowarray[i]->Packets.size()==100){

   if(!flowarray[i]->protof)
{
snprintf(RFlowID, sizeof(RFlowID), "%s-%s--%s-%s",flowarray[i]->daddr, flowarray[i]->saddr,flowarray[i]->dport,flowarray[i]->sport);
int position =-1;
for(int j = i+1; j < flowarray.size(); j++){
if (strstr(flowarray[j]->flowID,RFlowID)!=NULL)
{
position=j;
break;

}
}
if (position>0)
{
 flowarray[position]->protof=true;
flowarray[i]->protof=true;

 if (strcmp(flowarray[i]->proto,flowarray[position]->proto)!=0)
{
  if (flowarray[i]->score > flowarray[position]->score)
     {
        strncpy(flowarray[position]->proto,flowarray[i]->proto,32);
       if (flowarray[i]->isServer==0)
           flowarray[position]->isServer=1;
       else if (flowarray[i]->isServer==1)
           flowarray[position]->isServer=0;
     }
  else if (flowarray[i]->score < flowarray[position]->score)
     {
        strncpy(flowarray[i]->proto,flowarray[position]->proto,32);
       if (flowarray[position]->isServer==0)
           flowarray[i]->isServer=1;
       else if (flowarray[position]->isServer==1)
           flowarray[i]->isServer=0;
     }

}
else
{
  if (flowarray[i]->score > flowarray[position]->score)
     {
       if (flowarray[i]->isServer==0)
           flowarray[position]->isServer=1;
       else if (flowarray[i]->isServer==1)
           flowarray[position]->isServer=0;
     }
  else if (flowarray[i]->score < flowarray[position]->score)
     {
       if (flowarray[position]->isServer==0)
           flowarray[i]->isServer=1;
       else if (flowarray[position]->isServer==1)
           flowarray[i]->isServer=0;
     }

}
}
}
//validate the Bidirection flows 
   if(flowarray[i]->isServer==1&&flowarray[i]->specialType!=3){
            char new_proto[32];
            const char *type = "-S";
            strncpy (new_proto, flowarray[i]->proto,32);
            strncat (new_proto, type,32);
        strncpy(flowarray[i]->proto,new_proto,32);
    }
    else if (flowarray[i]->isServer==0&&flowarray[i]->specialType!=3){
        char new_proto[32];
        const char *type = "-C";
            strncpy (new_proto, flowarray[i]->proto,32);
            strncat (new_proto, type,32);
     if(strcmp(flowarray[i]->proto,"HTTP")==0)
{
 int pos;
 char ID[32];
 strncpy (ID, flowarray[i]->daddr,32);
 strncat (ID, flowarray[i]->dport,32);
 pos=FindService(services,ID);
if(pos>=0)
{
 GetURLs(services[pos],flowarray[i]->Packets);
}
}

        strncpy(flowarray[i]->proto,new_proto,32);
        
    }
    else if(flowarray[i]->specialType==3){
            strncpy(flowarray[i]->proto, "Unknown", 32);
                }
    test_counter++;
    }
}
 for (int j=0; j< services.size();j++)
{
   if(strcmp(services[j]->label,"HTTP")==0 && services[j]->URLS.size()>0)
{
std::string label=GetMSLabel(services[j]->URLS);
 if (!label.empty())
   services[j]->MSlabel.assign(label);
}

}
/*********************validate label***********************/ 
         // performance metrics calculation   
     printf("Collecting performance figures\n");
     int counter = 0;
     double diff, RST;
     
    if(log[0] != '*'){ // anything but '*' indicates that log should be used
        string log_str = "DyMonD/logs/";
        if(mode_buf[0] == 'i'){
            
            log_str.append("temp-log.txt");
        } else {
           
            log_str.append(log);
            
        }
        
        FP.open(log_str, std::ios_base::out); 

        
        printf("Writing to log\n");

     for(int i = 0; i < flowarray.size(); i++) {
         if (flowarray[i]->Packets.size() == 100 && strstr(flowarray[i]->proto,"Unknown") == NULL &&flowarray[i]->protof ) {
               if(strstr(flowarray[i]->proto,"HTTP") != NULL) {
           char SID[32];
           if (flowarray[i]->isServer==0)
              {
                strncpy (SID, flowarray[i]->daddr,32);
                strncat (SID, flowarray[i]->dport,32);
              }
            else if (flowarray[i]->isServer==1)
              {
                strncpy (SID, flowarray[i]->saddr,32);
                strncat (SID, flowarray[i]->sport,32);
              }
            int index=FindService(services,SID);
           if (index>=0)
             {
              std::string newproto= (std::string)flowarray[i]->proto;
              newproto.insert(4,services[index]->MSlabel);   
              strncpy(flowarray[i]->proto,newproto.c_str(),32);
  
            }
        }

            if (flowarray[i]->Ack_times.size() > 1 && flowarray[i]->isServer==1) {
                 diff = 0.0;
                 for (int j = 0; j < flowarray[i]->Ack_times.size(); j++) {
                     if (j != flowarray[i]->Ack_times.size() - 1)
                         diff += (flowarray[i]->Ack_times[j + 1]->sec +
                                  flowarray[i]->Ack_times[j + 1]->usec * 0.000001) -
                                 (flowarray[i]->Ack_times[j]->sec + flowarray[i]->Ack_times[j]->usec * 0.000001);
                 }
                 RST = (int)(abs(diff / (flowarray[i]->Ack_times.size() - 1)) * 1000.0)/1000.0;
                 FP << flowarray[i]->saddr << ":" << flowarray[i]->sport << " " << flowarray[i]->daddr << ":"
                    << flowarray[i]->dport <<" " << flowarray[i]->proto << " " << flowarray[i]->NumBytes / 30 << "-" << RST << "\n";
             } else {
                 FP << flowarray[i]->saddr << ":" << flowarray[i]->sport << " " << flowarray[i]->daddr << ":"
                    << flowarray[i]->dport  <<" " << flowarray[i]->proto << " " << flowarray[i]->NumBytes / 30 << "\n";
             }
         }
     }
     FP.close();
    if(argc == 1 || strstr(argv[1], "-t") != NULL) send_message(); // blank message indicates finished writing to log
    } else { // use tcp

        // For debugging flows
        FP.open(log_str, ios::app);

        for(int i = 0; i < flowarray.size(); i++) {
           if (flowarray[i]->Packets.size() == 100 && strstr(flowarray[i]->proto,"Unknown") == NULL &&flowarray[i]->protof) {
              if(strstr(flowarray[i]->proto,"HTTP") != NULL) {
           char SID[32];
           if (flowarray[i]->isServer==0)
              {
                strncpy (SID, flowarray[i]->daddr,32);
                strncat (SID, flowarray[i]->dport,32);
              }
            else if (flowarray[i]->isServer==1)
              {
                strncpy (SID, flowarray[i]->saddr,32);
                strncat (SID, flowarray[i]->sport,32);
              }
            int index=FindService(services,SID);
           if (index>=0)
             {
              std::string newproto= (std::string)flowarray[i]->proto;
              newproto.insert(4,services[index]->MSlabel);   
              strncpy(flowarray[i]->proto,newproto.c_str(),32);
  
            }
        }

               if(flowarray[i]->Ack_times.size()>1 && flowarray[i]->isServer==1){
                    diff=0.0;
                    for(int j = 0; j < flowarray[i]->Ack_times.size(); j++) {
                        if (j!=flowarray[i]->Ack_times.size()-1)
                            diff += (flowarray[i]->Ack_times[j + 1]->sec +
                                  flowarray[i]->Ack_times[j + 1]->usec * 0.000001) -
                                 (flowarray[i]->Ack_times[j]->sec + flowarray[i]->Ack_times[j]->usec * 0.000001);
                    }
                    RST = abs(diff/( flowarray[i]->Ack_times.size() -1)); 
                    add_to_flow_array(flowarray[i], RST, duration);

                    // For debugging flows
                    FP << flowarray[i]->saddr << ":" << flowarray[i]->sport << " " << flowarray[i]->daddr << ":"
                    << flowarray[i]->dport <<" " << flowarray[i]->proto << " " << flowarray[i]->NumBytes / 30 << "-" << RST << "\n";
                }
                else {
                    add_to_flow_array(flowarray[i], 0.0, duration);

                    // For debugging flows
                    FP << flowarray[i]->saddr << ":" << flowarray[i]->sport << " " << flowarray[i]->daddr << ":"
                    << flowarray[i]->dport  <<" " << flowarray[i]->proto << " " << flowarray[i]->NumBytes / 30 << "\n";
                }
                counter++;
           }
        }

        FP.close();
        if (!standalone) {
            send_message(flowarray);
            printf("Flows sent to controller\n");
            send_message();
        }
        counter = 0;
    }
     for(int i = 0; i < flowarray.size(); i++)
     {
        for (int j=0; j<flowarray[i]->Ack_times.size(); j++)
             free(flowarray[i]->Ack_times[j]);
         free(flowarray[i]);
     }
    }
        /*
        printf("Sniffing completed\n");
        clock_t start4 = clock();
        Py_DECREF(ArgList);
        Py_DECREF(PyList);
        Py_DECREF(pReturn);
        Py_DECREF(pFunc);
        Py_DECREF(pModule);
        Py_XDECREF(pDict);
        Py_Finalize(); 
        clock_t end4 = clock();        
        double elapsed4 = double(end4 - start4)/CLOCKS_PER_SEC;// environment finalization time */
    return 0;
}

