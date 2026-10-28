"""Patch-level read-only C guidance interventions; not retraining or F-aux."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
import torch.nn.functional as TF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.loss.boundary import semantic_boundary_target

WORK=Path('/hy-tmp/stage2_diagnostics_20261029')
EVAL=Path('/hy-tmp/stage2_eval_20261024')

def summary(x):
    a=x.detach().float().cpu().numpy()
    return {'min':float(a.min()),'max':float(a.max()),'mean':float(a.mean()),'std':float(a.std()),
            'fraction_above_0_5':float((a>.5).mean()),'fraction_above_0_9':float((a>.9).mean())}

def main():
    torch.set_num_threads(1);torch.manual_seed(1234);np.random.seed(1234)
    protocol=json.loads((EVAL/'protocol.json').read_text());model=protocol['models']['C']
    checkpoint=Path(model['checkpoint']);before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert before==model['checkpoint_sha256'] and torch.cuda.is_available()
    predictor=nnUNetPredictor(use_mirroring=False,device=torch.device('cuda'),allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(model['model_folder'],['all'],'checkpoint_best.pth')
    network=predictor.network;network.load_state_dict(predictor.list_of_parameters[0]);network.eval().cuda()
    del predictor.list_of_parameters
    preprocessor=predictor.configuration_manager.preprocessor_class(verbose=False)
    shape=np.array(predictor.configuration_manager.patch_size)
    reports=[]
    selections=[('FLARETs_0031',9),('FLARETs_0032',2),('FLARETs_0048',12),('FLARETs_0001',13),('FLARETs_0010',4)]
    for case,label in selections:
        row=next(r for r in protocol['cases'] if r['id']==case)
        data,seg,properties=preprocessor.run_case([row['image']],row['gt'],predictor.plans_manager,predictor.configuration_manager,predictor.dataset_json)
        points=np.argwhere(seg[0]==label)
        assert len(points)>0
        center=np.median(points,axis=0).astype(int);starts=np.maximum(0,np.minimum(center-shape//2,np.maximum(0,np.array(data.shape[1:])-shape)))
        slices=tuple(slice(int(s),int(s+n)) for s,n in zip(starts,shape))
        image=data[(slice(None),)+slices];target=seg[(slice(None),)+slices]
        pad=[(0,0)]+[(0,int(n-a)) for n,a in zip(shape,image.shape[1:])]
        image=np.pad(image,pad,constant_values=0);target=np.pad(target,pad,constant_values=-1)
        del data,seg,points
        x=torch.from_numpy(image[None].copy()).cuda();gt=torch.from_numpy(target[None].astype(np.int64)).cuda()
        boundary_target,mask=semantic_boundary_target(gt,ignore_label=-1)
        mode={'name':'natural'};trace={};maps={};head={};bottleneck={}
        def capture_head(module,args,output):head['logits']=output.detach().cpu()
        def capture_encoder(module,args,output):bottleneck['shape']=tuple(output[-1].shape[2:])
        handles=[network.boundary_head.register_forward_hook(capture_head),network.encoder.register_forward_hook(capture_encoder)]
        for index,agg in enumerate(network.mamba_mlp.aggregators):
            def hook(module,args,index=index):
                history,b=args
                if mode['name']=='zero':return (history,torch.zeros_like(b))
                if mode['name']=='mean':return (history,b.mean(dim=1,keepdim=True).expand_as(b))
                if mode['name']=='shuffle':
                    generator=torch.Generator(device=b.device).manual_seed(20261029)
                    return (history,b[:,torch.randperm(b.shape[1],generator=generator,device=b.device)])
                if mode['name']!='natural':return
                natural,w=module.forward(history,b,return_weights=True)
                zero,w0=module.forward(history,torch.zeros_like(b),return_weights=True)
                trace[str(index)]={'sources':len(history),'query_norm':float(module.query.norm()),
                    'boundary_query_norm':float(module.boundary_query.norm()),'weight_abs_difference_mean':float((w-w0).abs().mean()),
                    'weight_abs_difference_max':float((w-w0).abs().max()),'aggregate_relative_L2':float((natural.float()-zero.float()).norm()/(natural.float().norm()+1e-12)),
                    'natural_entropy_mean':float((-(w*w.clamp_min(1e-12).log()).sum(-1)).mean()),'prior':summary(b)}
                maps[str(index)]=(w.detach().cpu().numpy(),w0.detach().cpu().numpy())
                if index==0:head['prior']=b.detach().cpu()
            handles.append(agg.register_forward_pre_hook(hook))
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            natural=network(x).float().cpu();head_natural=head['logits'].clone();prior=head['prior'].clone()
            mode['name']='repeat';repeat=network(x).float().cpu()
            interventions={};predictions={'natural':natural.argmax(1).numpy()}
            for name in ['zero','mean','shuffle']:
                mode['name']=name;changed=network(x).float().cpu()
                interventions[name]={'logit_abs_difference_mean':float((natural-changed).abs().mean()),
                    'logit_abs_difference_max':float((natural-changed).abs().max()),
                    'relative_L2':float((natural-changed).norm()/(natural.norm()+1e-12)),
                    'label_changed_fraction':float((natural.argmax(1)!=changed.argmax(1)).float().mean())}
                predictions[name]=changed.argmax(1).numpy();del changed
        for h in handles:h.remove()
        probability=head_natural.float().sigmoid();bt=boundary_target.cpu();valid=mask.cpu()
        binary=probability>.5;boundary_dice=float((2*(binary*bt*valid).sum()+1e-8)/((binary*valid).sum()+(bt*valid).sum()+1e-8))
        pooled=prior.transpose(1,2).reshape(1,1,*bottleneck['shape'])
        up=TF.interpolate(pooled.float(),size=tuple(shape),mode='nearest')[0,0].numpy()
        z=int(bt[0,0].sum(dim=(1,2)).argmax())
        fig,axes=plt.subplots(2,3,figsize=(13,8))
        panels=[image[0,z],bt[0,0,z].numpy(),probability[0,0,z].numpy(),up[z],predictions['natural'][0,z],predictions['zero'][0,z]]
        titles=['Normalized CT','Semantic boundary target','Predicted boundary probability','Max-pooled prior (nearest upsample)','C natural prediction','C zero-guide prediction']
        for ax,array,title in zip(axes.flat,panels,titles):
            ax.imshow(array,cmap='gray' if title=='Normalized CT' else ('viridis' if 'probability' in title or 'prior' in title else 'tab20'))
            ax.set_title(title);ax.axis('off')
        fig.suptitle(case+' patch z='+str(z)+' (not full-volume evaluation)');fig.tight_layout();fig.savefig(WORK/(case+'_boundary.png'),dpi=140);plt.close(fig)
        fig,axes=plt.subplots(2,5,figsize=(17,5))
        arrays={'boundary_probability':probability.numpy(),'pooled_prior':pooled.numpy(),'boundary_target':bt.numpy(),'input_patch':image,'segmentation_target':target}
        for i,(w,w0) in enumerate(maps.values()):
            axes[0,i].imshow(w[0].T,aspect='auto',vmin=0,vmax=1);axes[1,i].imshow(np.abs(w-w0)[0].T,aspect='auto')
            axes[0,i].set_title('Aggregator '+str(i)+' natural');axes[1,i].set_title('|natural-zero|, same history')
            arrays['weights_'+str(i)]=w;arrays['weights_zero_'+str(i)]=w0
        fig.tight_layout();fig.savefig(WORK/(case+'_depth_weights.png'),dpi=140);plt.close(fig)
        np.savez_compressed(WORK/(case+'_patch_maps.npz'),**arrays)
        report={'case':case,'target_label_for_patch_selection':label,'patch_start_in_preprocessed_grid':starts.tolist(),'patch_shape':shape.tolist(),
            'preprocessed_crop_properties':{k:properties[k] for k in ['shape_before_cropping','shape_after_cropping_and_before_resampling','bbox_used_for_cropping']},
            'boundary_probability':summary(probability),'pooled_prior':summary(prior),'boundary_Dice_at_0_5_patch_only':boundary_dice,
            'GT_boundary_fraction':float(bt.float().mean()),'natural_repeat_max_abs_difference':float((natural-repeat).abs().max()),
            'aggregators_same_history_comparison':trace,'end_to_end_patch_interventions':interventions}
        reports.append(report);(WORK/'guidance_progress.json').write_text(json.dumps(reports,indent=2)+'\n')
        print('GUIDANCE_CASE_COMPLETED',case,json.dumps(interventions),flush=True)
        del x,gt,natural,repeat,head,head_natural,probability,prior,pooled,arrays,maps
        torch.cuda.empty_cache()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==before
    (WORK/'guidance_summary.json').write_text(json.dumps({'status':'completed','checkpoint_sha256_before_and_after':before,
        'scope':'Five GT-targeted diagnostic patches, not full cases or an unbiased performance estimate; no tuning or training',
        'same_history_comparison':'Isolates immediate conditioning effect; end-to-end interventions also change later histories',
        'not_equivalent_to_F_aux':True,'reports':reports},indent=2)+'\n')

if __name__=='__main__':main()
