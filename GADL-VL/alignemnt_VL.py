from algorithm import *
# from graphMatching import *
from networkx import read_edgelist
from scipy.io import loadmat
from model_VL import *
from utils import *
import scipy.io as io
import dgl
from torch_geometric.datasets import DBP15K
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.utils import dense_to_sparse
from torch.optim import Adam
import argparse
from load_data_vision import load_vision_language, load_config


    
def run_GADL(GAE_model, train_adj, train_features, test_pairs, args, device):

    optimizer = Adam(GAE_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    
    keys = train_adj.keys()
    A1 = train_adj[list(keys)[0]][0]
    X1 = train_features[list(keys)[0]][0]
    A2 = train_adj[list(keys)[1]][0]
    X2 = train_features[list(keys)[1]][0]
    
    X1 = X1.to(device)
    X2 = X2.to(device)
            

    L1, lam1, phi1 = compute_laplacian_eigenbasis(A1, args.k, norm_type='sym')
    L2, lam2, phi2 = compute_laplacian_eigenbasis(A2, args.k, norm_type='sym')

    phi1, lam1 = phi1.to(device), lam1.to(device)
    phi2, lam2 = phi2.to(device), lam2.to(device)
    
    ## Precompute eigendecomposition of augmented adjacency matrix
    lam_A1, phi_A1 = compute_eigen_decomposition(A1, mode='sym', add_self_loops=True, k=args.k)
    lam_A2, phi_A2 = compute_eigen_decomposition(A2, mode='sym', add_self_loops=True, k=args.k)
    phi_A1, lam_A1 = phi_A1.to(device), lam_A1.to(device)
    phi_A2, lam_A2 = phi_A2.to(device), lam_A2.to(device)

    # Compute descriptors using HKS
    desc1 = compute_hks(phi1, lam1)
    desc2 = compute_hks(phi2, lam2)
    
    best_result =0
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=10, verbose=True)
        
    with tqdm(total=args.epoch, desc='(T)') as pbar:
        for step in range(1,args.epoch+1):

            GAE_model.train()

            adj1 = coo_matrix(A1.numpy())
            adj2 = coo_matrix(A2.numpy())

            pos_weight1 = float(adj1.shape[0] * adj1.shape[0] - adj1.sum()) / adj1.sum()
            pos_weight2 = float(adj2.shape[0] * adj2.shape[0] - adj2.sum()) / adj2.sum()
            norm1 = adj1.shape[0] * adj1.shape[0] / float((adj1.shape[0] * adj1.shape[0] - adj1.sum()) * 2)
            norm2 = adj2.shape[0] * adj2.shape[0] / float((adj2.shape[0] * adj2.shape[0] - adj2.sum()) * 2)


            # GCN-style normalization \hat{A} = \hat{D}^{-1/2} (A+I) \hat{D}^{-1/2}
            # adj_norm1 = preprocess_graph(adj1)
            # adj_norm2 = preprocess_graph(adj2)
            
            # adj_norm1 = torch.sparse.FloatTensor(torch.LongTensor(adj_norm1[0].T), torch.FloatTensor(adj_norm1[1]),torch.Size(adj_norm1[2])).to(device)
            # adj_norm2 = torch.sparse.FloatTensor(torch.LongTensor(adj_norm2[0].T), torch.FloatTensor(adj_norm2[1]),torch.Size(adj_norm2[2])).to(device)
            
            adj_label1 = sparse_to_tuple(adj1)
            adj_label2 = sparse_to_tuple(adj2)
            adj_label1 = torch.sparse.FloatTensor(torch.LongTensor(adj_label1[0].T), torch.FloatTensor(adj_label1[1]),torch.Size(adj_label1[2])).to(device)
            adj_label2 = torch.sparse.FloatTensor(torch.LongTensor(adj_label2[0].T), torch.FloatTensor(adj_label2[1]), torch.Size(adj_label2[2])).to(device)


            weight_mask1 = adj_label1.to_dense().view(-1) == 1
            weight_mask2 = adj_label2.to_dense().view(-1) == 1
            weight_tensor1 = torch.ones(weight_mask1.size(0))
            weight_tensor2 = torch.ones(weight_mask2.size(0))
            weight_tensor1[weight_mask1] = pos_weight1
            weight_tensor2[weight_mask2] = pos_weight2
            weight_tensor1 = weight_tensor1.to(device)
            weight_tensor2 = weight_tensor2.to(device)
            
            adj_norm1 = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj1), device)
            adj_norm2 = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj2), device)
              
            z1_h, z1_l, z2_h, z2_l = GAE_model(X1, adj1, X2, adj2)      

            z1 = torch.cat([z1_l, z1_h],dim=1)
            z2 = torch.cat([z2_l, z2_h],dim=1)
                        
            if args.LS_alignemnt:
                loss_fm, loss_map = GAE_model.functional_map(z1, z2, desc1, desc2, phi1, phi2, lam1, lam2)           
            else:
                loss_fm, loss_map = 0, 0
            
            adj_pred1 = torch.sigmoid(torch.matmul(z1, z1.t()))
            adj_pred2 = torch.sigmoid(torch.matmul(z2, z2.t()))
            
            rec_loss1 = norm1 * F.binary_cross_entropy(adj_pred1.view(-1), adj_label1.to_dense().view(-1), weight=weight_tensor1)
            rec_loss2 = norm2 * F.binary_cross_entropy(adj_pred2.view(-1), adj_label2.to_dense().view(-1), weight=weight_tensor2)
            

            l_FM = max(1.0 * (1 - step / args.epoch), 0.5)  # Decrease FM weight over time                  
            
            loss = rec_loss1 + rec_loss2 + l_FM *loss_fm + loss_map
   
    
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(GAE_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            ## evaluation
            if step%5 or step==args.epoch:
                GAE_model.eval()
                with torch.no_grad():
                    eval_z1_h, eval_z1_l, eval_z2_h, eval_z2_l = GAE_model(X1.to(device), adj1, X2.to(device), adj2)
                    
                    # Combine embeddings
                    eval_z1_combined = torch.cat([eval_z1_l, eval_z1_h], dim=1)
                    eval_z2_combined = torch.cat([eval_z2_l, eval_z2_h], dim=1)
                    
                    # eval_z1_combined = eval_z1_l
                    # eval_z2_combined = eval_z2_l
                    
                    eval_z1_norm = F.normalize(eval_z1_combined, p=2, dim=1)
                    eval_z2_norm = F.normalize(eval_z2_combined, p=2, dim=1)
                    
                    
                    # Distance-based matching
                    D = torch.cdist(eval_z1_norm.double(), eval_z2_norm.double(), 2)
                    P_HG = get_match(D, device)
                    c = 0
                    P = torch.eye(eval_z1_norm.shape[0])
                    for j in range(P_HG.size(0)):
                        r1 = P_HG[j].cpu()
                        r2 = P[j].cpu()
                        if r1.equal(r2): 
                            c += 1
                    
                    cur_result = (c / eval_z1_norm.shape[0]) * 100

                    if cur_result > best_result:
                        best_result = cur_result
                    
                    scheduler.step(cur_result)
                    
            
            pbar.update()
            
            # Early stopping based on learning rate
            if optimizer.param_groups[0]['lr'] < 1e-6:
                print(f"Early stopping at epoch {step} due to low learning rate")
                break
            

        
    # print("\n" + "="*50)
    # print("FINAL RESULTS:")
    # print("="*50)
    # print(f"Best Distance-based Accuracy: {best_result:.2f}%")
    return best_result
            
         

def main(args):
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")

    print("Loading training datasets")       
    train_features, train_adj, test_pairs, modals_name = load_vision_language(args.dataset, args.vision, args.language, args.k_neig, portion=args.portion)        
    vision_n = train_features['vision'][0].shape[0]
    vision_dim = train_features['vision'][0].shape[1]
    language_dim = train_features['language'][0].shape[1]                
    
    args.k = vision_n-1 if args.k > vision_n else args.k
       
    print("Fitting model")
    
    best=[]
    for run_idx in range(args.runs):
        # Set different random seed for each run
        torch.manual_seed(42 + run_idx)
        np.random.seed(42 + run_idx)
    
        gconv1 = GAE_GConv_vl(args.num_hidden_layers, vision_dim, args.hidden_dim, args.output_feature_size).to(device)
        gconv2 = GAE_GConv_vl(args.num_hidden_layers, language_dim, args.hidden_dim, args.output_feature_size).to(device)

        encoder_model = Encoder_GAE_FM(encoder1=gconv1, encoder2=gconv2, hidden_dim= args.hidden_dim, k = args.k).to(device)    

        print(f"\n{'='*50}")
        print(f"Run {run_idx + 1}/{args.runs}")
        print(f"{'='*50}")
    
        result = run_GADL(encoder_model, train_adj, train_features, test_pairs, args, device)
        best.append(result)
        print(f"Run {run_idx + 1} Result: {result:.2f}%")

    mean = np.mean(best)
    std = np.std(best)  
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print("="*50)
    print(f'vision model: {args.vision}')
    print(f'language model: {args.language}')
    print(f'final results:{mean:.2f} ± {std:.2f}')
    return f'{mean:.2f} ± {std:.2f}'
    

def parse_args():
    parser = argparse.ArgumentParser(description="Run GADL for vision-language alignemnt task")
    parser.add_argument('--device',type=str, help='GPU_id', default='1')
    parser.add_argument('--dataset', type=str, default="CIFAR-100",  choices=["CINIC-10", "CIFAR-10", "CIFAR-100", "Imagenet-100"]) 
    parser.add_argument('--LS_alignemnt', type=bool, help="With latent-space alignemnt", default=True)
    # Dataset-specific placeholders (optional override)
    parser.add_argument('--k', type=int, help='number of basis functions')
    parser.add_argument('--k_neig', type=int, help='number of neighbors in constructing graph')
    parser.add_argument('--input_dim', type=int)
    parser.add_argument('--portion', type=float, help='portaion on classes used: 1:all', default=1)
    parser.add_argument('--num_hidden_layers', type=int)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--output_feature_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float, default=5e-4) 
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--encoder', type=str)
    parser.add_argument('--vision', type=str) 
    parser.add_argument('--language', type=str) 
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args, config = load_config("config.yaml", args.dataset, args)
    
    values = main(args)   
    