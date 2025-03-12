
<html>

<head>
<meta http-equiv=Content-Type content="text/html; charset=utf-8">
<meta name=Generator content="Microsoft Word 15 (filtered)">
<style>
<!--
 /* Font Definitions */
 @font-face
	{font-family:Helvetica;
	panose-1:0 0 0 0 0 0 0 0 0 0;}
@font-face
	{font-family:"Cambria Math";
	panose-1:2 4 5 3 5 4 6 3 2 4;}
@font-face
	{font-family:DengXian;
	panose-1:2 1 6 0 3 1 1 1 1 1;}
@font-face
	{font-family:"Palatino Linotype";
	panose-1:2 4 5 2 5 5 5 3 3 4;}
@font-face
	{font-family:"\@等线";
	panose-1:2 1 6 0 3 1 1 1 1 1;}
 /* Style Definitions */
 p.MsoNormal, li.MsoNormal, div.MsoNormal
	{margin:0cm;
	text-align:justify;
	text-justify:inter-ideograph;
	font-size:10.5pt;
	font-family:DengXian;}
a:link, span.MsoHyperlink
	{color:#0563C1;
	text-decoration:underline;}
 /* Page Definitions */
 @page WordSection1
	{size:595.3pt 841.9pt;
	margin:72.0pt 90.0pt 72.0pt 90.0pt;
	layout-grid:15.6pt;}
div.WordSection1
	{page:WordSection1;}
-->
</style>

</head>

<body lang=ZH-CN link="#0563C1" vlink="#954F72" style='word-wrap:break-word;
text-justify-trim:punctuation'>

<div class=WordSection1 style='layout-grid:15.6pt'>

<p class=MsoNormal><span lang=EN-US>&nbsp;</span></p>

<p class=MsoNormal><span lang=EN-US>&nbsp;</span></p>

<p class=MsoNormal align=center style='text-align:center;line-height:30.0pt'><b><span
lang=EN-US style='font-size:36.0pt;font-family:"Palatino Linotype",serif'>Memory
based Temporal Fusion Network for Video Deblurring</span></b></p>

<p class=MsoNormal align=center style='text-align:center;line-height:45.0pt'><span
lang=EN-US><a href="https://github.com/fengzhuziA"><span style='font-size:22.0pt;
font-family:"Palatino Linotype",serif'>Chaohua Wang</span></a></span><sup><span
lang=EN-US style='font-size:22.0pt;font-family:"Palatino Linotype",serif;
color:#0070C0'>1</span></sup><span lang=EN-US style='font-size:22.0pt;
font-family:"Palatino Linotype",serif;color:#0070C0'>       </span><span
lang=EN-US><a href="https://see.xidian.edu.cn/faculty/wsdong/"><span
style='font-size:22.0pt;font-family:"Palatino Linotype",serif'>Weisheng Dong</span></a></span><sup><span
lang=EN-US style='font-size:22.0pt;font-family:"Palatino Linotype",serif;
color:#0070C0'>1</span></sup><span lang=EN-US style='font-size:22.0pt;
font-family:"Palatino Linotype",serif;color:#0070C0'>*  </span></p>

<p class=MsoNormal align=center style='text-align:center;line-height:45.0pt'><span
lang=EN-US style='font-size:22.0pt;font-family:"Palatino Linotype",serif;
color:#0070C0'>Xin Li<sup>2      </sup>Fangfang Wu<sup>1</sup>   </span><span
lang=EN-US><a href="https://web.xidian.edu.cn/wjj/"><span style='font-size:
22.0pt;font-family:"Palatino Linotype",serif'>Jinjian Wu</span></a></span><sup><span
lang=EN-US style='font-size:22.0pt;font-family:"Palatino Linotype",serif;
color:#0070C0'>1</span></sup><span lang=EN-US style='font-size:22.0pt;
font-family:"Palatino Linotype",serif;color:#0070C0'>       </span><span
lang=EN-US><a href="https://web.xidian.edu.cn/gmshi/"><span style='font-size:
22.0pt;font-family:"Palatino Linotype",serif'>Guangming Shi</span></a></span><sup><span
lang=EN-US style='font-size:22.0pt;font-family:"Palatino Linotype",serif;
color:#0070C0'>1</span></sup></p>

<p class=MsoNormal align=center style='text-align:center;line-height:45.0pt'><sup><span
lang=EN-US style='font-size:22.0pt;font-family:"Palatino Linotype",serif;
color:#0070C0'>1</span></sup><span lang=EN-US style='font-size:22.0pt;
font-family:"Palatino Linotype",serif;color:#0070C0'>School of Artificial
Intelligence, Xidian University                   <sup>2</sup>West Virginia
University</span></p>

<p class=MsoNormal align=center style='text-align:center;line-height:45.0pt'><span
lang=EN-US style='font-size:20.0pt;font-family:"Palatino Linotype",serif;
color:#0070C0'>&nbsp;</span></p>

<p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
style='font-size:18.0pt;font-family:"Palatino Linotype",serif'><img border=0
width=789 height=572 id="图片 2" src="TFNet.fld/image001.png"></span></p>

<p class=MsoNormal align=center style='text-align:center;line-height:30.0pt'><span
lang=EN-US style='font-size:18.0pt;font-family:"Palatino Linotype",serif'>Figure
1. Architecture of the proposed network TFNet for video deblurring. </span></p>

<p class=MsoNormal align=center style='text-align:center;line-height:30.0pt'><span
lang=EN-US style='font-size:18.0pt;font-family:"Palatino Linotype",serif'>(a) The
architecture of the proposed network. (b) The architecture of the encoder. (c) The
architecture of the temporal fusion module. (d) The detailed architecture of
our local spatial-temporal memory based temporal fusion module.</span></p>

<p class=MsoNormal style='text-indent:22.0pt'><b><span lang=EN-US
style='font-size:22.0pt;font-family:"Palatino Linotype",serif'>Abstract</span></b></p>

<p class=MsoNormal><span lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'>Video
deblurring is one of the most challenging vision tasks because of the complex
spatial-temporal relationship and a number of uncertainty factors involved in
video acquisition. As different moving objects in the video exhibit different
motion trajectories, it is difficult to accurately capture their
spatial-temporal relationships. In this paper, we proposed a memory-based
temporal fusion network (TFN) to capture local spatial-temporal relationships
across the input sequence for video deblurring. Our temporal fusion network
consists of a memory network and a temporal fusion block. The memory network
stores the extracted spatial-temporal relationships and guides the temporal
fusion blocks to extract local spatial-temporal relationships more accurately.
In addition, to enable our model to more effectively fuse the multi-scale
features of the previous frame, we propose a multi-scale and multihop
reconstruction memory network (RMN) based on the attention mechanism and memory
network. We constructed a feature extractor that integrates residual dense
blocks with three downsample layers to extract hierarchical spatial features.
Finally, we feed these aggregated local features into a reconstruction module
to restore sharp video frames. Experimental results on public datasets show
that our temporal fusion network has achieved a significant performance
improvement in terms of PSNR metrics (over $1 dB$) over existing
state-of-the-art video deblurring methods.</span></p>

<p class=MsoNormal><span lang=EN-US style='font-size:12.0pt;font-family:"Times New Roman",serif;
color:#0070C0'>&nbsp;</span></p>

<p class=MsoNormal style='text-indent:22.0pt'><a name="OLE_LINK1"><b><span
lang=EN-US style='font-size:22.0pt;font-family:"Palatino Linotype",serif'>Paper</span></b></a></p>

<p class=MsoNormal>

<table cellpadding=0 cellspacing=0>
 <tr>
  <td width=415 height=0></td>
 </tr>
 <tr>
  <td></td>
  <td><img width=100 height=101 src="TFNet.fld/image002.png"></td>
 </tr>
</table>

<br clear=ALL>
<b><span lang=EN-US style='font-size:22.0pt;font-family:"Palatino Linotype",serif'>                         
              </span></b><span lang=EN-US><a
href="https://doi.org/10.1007/s11263-023-01793-y"><span style='font-size:16.0pt;
font-family:"Palatino Linotype",serif'>IJCV  2023 </span></a></span><span
lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'>                               </span></p>

<p class=MsoNormal><span lang=EN-US style='font-size:12.0pt;font-family:"Palatino Linotype",serif;
color:#0070C0'>&nbsp;</span></p>

<p class=MsoNormal style='text-indent:28.0pt'><b><i><span lang=EN-US
style='font-size:14.0pt;font-family:"Palatino Linotype",serif'>Citation</span></i></b></p>

<p class=MsoNormal style='margin-left:42.4pt;text-indent:.15pt'><span
lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'>C. Wang,
W. Dong, X. Li, F. Wu, J. Wu and G. Shi, &quot;</span><span lang=EN-US
style='font-size:15.5pt;font-family:Helvetica;color:black'> </span><span
lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'>Memory
based Temporal Fusion Network for Video Deblurring,&quot; in </span><span
lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'>International</span></p>

<p class=MsoNormal style='margin-left:42.4pt;text-indent:.15pt'><span
lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'>Journal
of Computer Vision (IJCV), doi: 10.1007/s11263-023-01793-y.</span></p>

<p class=MsoNormal style='margin-left:42.4pt;text-indent:.15pt'><span
lang=EN-US style='font-size:12.0pt;font-family:"Palatino Linotype",serif;
color:#0070C0'>&nbsp;</span></p>

<p class=MsoNormal style='text-indent:28.0pt'><b><i><span lang=EN-US
style='font-size:14.0pt;font-family:"Palatino Linotype",serif'>Bibtex</span></i></b></p>

<p class=MsoNormal style='margin-left:21.2pt;text-indent:20.95pt'><span
lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'>@article{wang2023memory,</span></p>

<p class=MsoNormal style='margin-left:21.2pt;text-indent:20.95pt'><span
lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'>  title={Memory
Based Temporal Fusion Network for Video Deblurring},</span></p>

<p class=MsoNormal style='margin-left:21.2pt;text-indent:20.95pt'><span
lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'> 
author={Wang, Chaohua and Dong, Weisheng and Li, Xin and Wu, Fangfang and Wu,
Jinjian and Shi, Guangming},</span></p>

<p class=MsoNormal style='margin-left:21.2pt;text-indent:20.95pt'><span
lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'> 
journal={International Journal of Computer Vision},</span></p>

<p class=MsoNormal style='margin-left:21.2pt;text-indent:20.95pt'><span
lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'> 
pages={1--17},</span></p>

<p class=MsoNormal style='margin-left:21.2pt;text-indent:20.95pt'><span
lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'> 
year={2023},</span></p>

<p class=MsoNormal style='margin-left:21.2pt;text-indent:20.95pt'><span
lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'> 
publisher={Springer}</span></p>

<p class=MsoNormal style='margin-left:21.2pt;text-indent:20.95pt'><span
lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'>}</span></p>

<p class=MsoNormal style='margin-left:21.2pt;text-indent:20.95pt'><span
lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'>&nbsp;</span></p>

<p class=MsoNormal style='line-height:12.0pt'><span lang=EN-US
style='font-size:12.0pt;font-family:"Times New Roman",serif;color:#0070C0'>&nbsp;</span></p>

<p class=MsoNormal style='line-height:12.0pt'><span lang=EN-US
style='font-size:12.0pt;font-family:"Times New Roman",serif;color:#0070C0'>&nbsp;</span></p>

<p class=MsoNormal style='line-height:12.0pt'><span lang=EN-US
style='font-size:12.0pt;font-family:"Times New Roman",serif;color:#0070C0'>&nbsp;</span></p>

<p class=MsoNormal style='line-height:12.0pt'><span lang=EN-US
style='font-size:12.0pt;font-family:"Times New Roman",serif;color:#0070C0'>&nbsp;</span></p>

<p class=MsoNormal style='text-indent:22.0pt'><b><span lang=EN-US
style='font-size:22.0pt;font-family:"Palatino Linotype",serif'>Download</span></b></p>

<p class=MsoNormal style='line-height:12.0pt'><span lang=EN-US
style='font-size:12.0pt;font-family:"Times New Roman",serif;color:#0070C0'>&nbsp;</span></p>

<p class=MsoNormal style='line-height:12.0pt'><span lang=EN-US
style='font-size:12.0pt;font-family:"Times New Roman",serif;color:#0070C0'>&nbsp;</span></p>

<p class=MsoNormal style='line-height:30.0pt'>

<table cellpadding=0 cellspacing=0>
 <tr>
  <td width=432 height=0></td>
 </tr>
 <tr>
  <td></td>
  <td><img width=85 height=85 src="TFNet.fld/image003.png"></td>
 </tr>
</table>

<br clear=ALL>
<span lang=EN-US style='font-size:12.0pt;font-family:"Times New Roman",serif;
color:#0070C0'>                        </span><span lang=EN-US
style='font-size:16.0pt;font-family:"Times New Roman",serif;color:#0070C0'>                                       </span><span
lang=EN-US style='font-size:14.0pt;font-family:"Palatino Linotype",serif'> <span
class=MsoHyperlink><a href="https://github.com/fengzhuziA/TFmodel">Code</a></span>      
                                </span></p>

<p class=MsoNormal style='line-height:12.0pt'><span lang=EN-US
style='font-size:11.0pt;font-family:"Times New Roman",serif;color:#0070C0'>&nbsp;</span></p>

<p class=MsoNormal style='line-height:12.0pt'><span lang=EN-US
style='font-size:12.0pt;font-family:"Times New Roman",serif;color:#0070C0'>&nbsp;</span></p>

<p class=MsoNormal style='line-height:12.0pt'><span lang=EN-US
style='font-size:12.0pt;font-family:"Times New Roman",serif;color:#0070C0'>  </span></p>

<p class=MsoNormal style='text-indent:22.0pt'><b><span lang=EN-US
style='font-size:22.0pt;font-family:"Palatino Linotype",serif'>Results:</span></b><span
class=MsoHyperlink><b><i><span lang=EN-US style='font-size:18.0pt;font-family:
"Palatino Linotype",serif;text-decoration:none'> Comparison with
State-of-the-art Reconstruction Methods:</span></i></b></span></p>

<p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
style='font-size:18.0pt;font-family:"Palatino Linotype",serif'>Table 1 and
Table 2 . The quantitative results on GOPRO and BSD dataset.</span></p>

<p class=MsoNormal align=center style='text-align:center'><b><span lang=EN-US
style='font-size:36.0pt;font-family:"Palatino Linotype",serif'><img border=0
width=1055 height=290 id="图片 3" src="TFNet.fld/image004.png"></span></b></p>

<p class=MsoNormal><span lang=EN-US style='font-size:20.0pt;font-family:"Palatino Linotype",serif'>&nbsp;</span></p>

<p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
style='font-size:20.0pt;font-family:"Palatino Linotype",serif'>&nbsp;</span></p>

<p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
style='font-size:20.0pt;font-family:"Palatino Linotype",serif'><img border=0
width=783 height=708 src="TFNet.fld/image005.png"></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:42.65pt;margin-bottom:
0cm;margin-left:35.4pt;margin-bottom:.0001pt;line-height:30.0pt'><span
lang=EN-US style='font-size:20.0pt;font-family:"Palatino Linotype",serif'>Figure
2. Visualizations of attention maps. (a) The input blurred frames. (b)
Deblurred frames by the proposed method. (c) The ground truth frames. (d)
Attention maps of the middle frame in adjacent frames.</span></p>

<p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
style='font-size:20.0pt;font-family:"Palatino Linotype",serif'><img border=0
width=1042 height=382 id="图片 1" src="TFNet.fld/image006.png"></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:42.65pt;margin-bottom:
0cm;margin-left:35.4pt;margin-bottom:.0001pt;line-height:30.0pt'><span
lang=EN-US style='font-size:20.0pt;font-family:"Palatino Linotype",serif'>Figure
3. The figure shows the two video clips A and B with the lowest and highest
PSNR scores in the GOPRO dataset.</span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:42.65pt;margin-bottom:
0cm;margin-left:35.4pt;margin-bottom:.0001pt'><span lang=EN-US
style='font-size:20.0pt;font-family:"Palatino Linotype",serif'>&nbsp;</span></p>

<p class=MsoNormal style='line-height:12.0pt'><span lang=EN-US
style='font-size:12.0pt;font-family:"Times New Roman",serif;color:#0070C0'>&nbsp;</span></p>

<p class=MsoNormal style='line-height:12.0pt'><span lang=EN-US
style='font-size:12.0pt;font-family:"Times New Roman",serif;color:#0070C0'>&nbsp;</span></p>

<p class=MsoNormal style='line-height:12.0pt'><span lang=EN-US
style='font-size:12.0pt;font-family:"Times New Roman",serif;color:#0070C0'>&nbsp;</span></p>

<p class=MsoNormal style='text-indent:22.0pt'><b><span lang=EN-US
style='font-size:22.0pt;font-family:"Palatino Linotype",serif'>References</span></b></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:49.75pt;margin-bottom:
0cm;margin-left:70.9pt;margin-bottom:.0001pt;text-indent:-22.4pt;line-height:
22.0pt'><span lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'>[1]</span><span
lang=EN-US style='font-size:8.0pt'> </span><span lang=EN-US style='font-size:
16.0pt;font-family:"Palatino Linotype",serif'>Wang X, Chan KCK, Yu K, Dong C,
Loy CC (2019) EDVR: video restoration with enhanced deformable convolutional
networks. In:IEEE Conference on ComputerVision and Pattern Recognition
Workshops, CVPR Workshops 2019, Long Beach, CA, USA, June 16-20, 2019,Computer
Vision Foundation / IEEE, pp 1954–196</span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:49.75pt;margin-bottom:
0cm;margin-left:70.9pt;margin-bottom:.0001pt;text-indent:-22.4pt;line-height:
22.0pt'><span lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'>[2]
Zhou S, Zhang J, Pan J, Zuo W, Xie H, Ren JSJ (2019) Spatio-temporal filter
adaptive network for video deblurring. In: 2019 IEEE/CVF International
Conference on Computer Vision, ICCV 2019, Seoul, Korea (South), October 27 -
November 2, 2019, IEEE, pp 2482–2491</span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:49.75pt;margin-bottom:
0cm;margin-left:70.9pt;margin-bottom:.0001pt;text-indent:-22.4pt;line-height:
22.0pt'><span lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'>[3]
Zhong Z, Gao Y, Zheng Y, Zheng B (2020) Efficient spatio-temporal recurrent
neural network for video deblurring. In: Vedaldi A, Bischof H, Brox T, Frahm J
(eds) Computer Vision - ECCV 2020 - 16th European Conference, Glasgow, UK,
August 23-28, 2020, Proceedings, Part VI, Springer, Lecture Notes in Computer
Science, vol 12351,pp 191–207</span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:49.75pt;margin-bottom:
0cm;margin-left:70.9pt;margin-bottom:.0001pt;text-indent:-22.4pt;line-height:
22.0pt'><span lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'>[4]
Zhang H, Dai Y, Li H, Koniusz P (2019) Deep stacked hierarchical multi-patch
network for image deblurring. In:IEEE Conference on Computer Vision and Pattern
Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, Computer Vision
Foundation / IEEE, pp 5978–5986</span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:49.75pt;margin-bottom:
0cm;margin-left:70.9pt;margin-bottom:.0001pt;text-indent:-22.4pt;line-height:
22.0pt'><span lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'>[5]
Tsai F, Peng Y, Lin Y, Tsai C, Lin C (2021) Banet: Bluraware attention networks
for dynamic scene deblurring. CoRR abs/2101.07518</span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:49.75pt;margin-bottom:
0cm;margin-left:70.9pt;margin-bottom:.0001pt;text-indent:-22.4pt;line-height:
22.0pt'><span lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'>[6]
Zamir SW, Arora A, Khan SH, Hayat M, Khan FS, Yang M, Shao L (2021) Multi-stage
progressive image restoration. In: IEEE Conference on Computer Vision and Pattern
Recognition, CVPR 2021, virtual, June 19-25, 2021,Computer Vision Foundation /
IEEE, pp 14821–14831</span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:49.75pt;margin-bottom:
0cm;margin-left:70.9pt;margin-bottom:.0001pt;text-indent:-22.4pt;line-height:
22.0pt'><span lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'>[7]
Chen L, Lu X, Zhang J, Chu X, Chen C (2021) Hinet: Halfinstance normalization
network for image restoration. In:IEEE Conference on Computer Vision and
Pattern Recognition Workshops, CVPR Workshops 2021, virtual, June19-25, 2021,
Computer Vision Foundation / IEEE, pp 182–192</span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:49.75pt;margin-bottom:
0cm;margin-left:70.9pt;margin-bottom:.0001pt;text-indent:-22.4pt;line-height:
22.0pt'><span lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'>[8]
Zhu C, Dong H, Pan J, Liang B, Huang Y, Fu L, Wang F (2021) Deep recurrent
neural network with multi-scale bi-directional propagation for video
deblurring. CoRR abs/2112.05150</span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:49.75pt;margin-bottom:
0cm;margin-left:70.9pt;margin-bottom:.0001pt;text-indent:-22.4pt;line-height:
22.0pt'><span lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'>[9]
Kim TH, Lee KM, Sch¨ olkopf B, Hirsch M (2017) Online video deblurring via
dynamic temporal blending network.In: IEEE International Conference on Computer
Vision,ICCV 2017, Venice, Italy, October 22-29, 2017, IEEE Computer Society, pp
4058–4067</span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:49.75pt;margin-bottom:
0cm;margin-left:70.9pt;margin-bottom:.0001pt;text-indent:-22.4pt;line-height:
22.0pt'><span lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'>[10]
Su S, Delbracio M, Wang J, Sapiro G, Heidrich W, Wang O (2017) Deep video
deblurring for hand-held cameras. In:2017 IEEE Conference on Computer Vision
and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, IEEE Computer
Society, pp 237–246</span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:49.75pt;margin-bottom:
0cm;margin-left:70.9pt;margin-bottom:.0001pt;text-indent:-22.4pt;line-height:
22.0pt'><span lang=EN-US style='font-size:16.0pt;font-family:"Palatino Linotype",serif'>[11]
Nah S, Son S, Lee KM (2019) Recurrent neural networks with intra-frame
iterations for video deblurring. In: IEEEConference on Computer Vision and
Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019,Computer
Vision Foundation / IEEE, pp 8102–8111</span></p>

<p class=MsoNormal style='line-height:12.0pt'><span lang=EN-US
style='font-size:12.0pt;font-family:"Times New Roman",serif;color:#0070C0'>&nbsp;</span></p>

<p class=MsoNormal style='line-height:12.0pt'><span lang=EN-US
style='font-size:12.0pt;font-family:"Times New Roman",serif;color:#0070C0'>&nbsp;</span></p>

<p class=MsoNormal style='text-indent:22.0pt'><b><span lang=EN-US
style='font-size:22.0pt;font-family:"Palatino Linotype",serif'>Contact</span></b></p>

<p class=MsoNormal style='text-indent:72.0pt;line-height:28.0pt'><span
lang=EN-US style='font-size:18.0pt;font-family:"Palatino Linotype",serif'>Chaohua
Wang, Email: <a href="mailto:3267928656@qq.com">3267928656@qq.com</a></span></p>

<p class=MsoNormal style='text-indent:72.0pt;line-height:28.0pt'><span
lang=EN-US style='font-size:18.0pt;font-family:"Palatino Linotype",serif'>Weisheng
Dong, Email: wsdong@mail.xidian.edu.cn</span></p>

<p class=MsoNormal style='text-indent:72.0pt;line-height:28.0pt'><span
lang=EN-US style='font-size:18.0pt;font-family:"Palatino Linotype",serif'>Xin Li,
Email: <a href="mailto:xin.li@ieee.org">xin.li@ieee.org</a></span></p>

<p class=MsoNormal style='text-indent:72.0pt;line-height:28.0pt'><span
lang=EN-US style='font-size:18.0pt;font-family:"Palatino Linotype",serif'>FangFang
Wu, Email:</span><span lang=EN-US style='font-size:8.0pt;font-family:Helvetica;
color:black'> </span><span lang=EN-US style='font-size:18.0pt;font-family:"Palatino Linotype",serif'>ffwu
<a href="mailto:xd@163.com">xd@163.com</a></span></p>

<p class=MsoNormal style='text-indent:72.0pt;line-height:28.0pt'><span
lang=EN-US style='font-size:18.0pt;font-family:"Palatino Linotype",serif'>Jinjian
Wu,Email:</span><span lang=EN-US style='font-size:8.0pt;font-family:Helvetica;
color:black'> </span><span lang=EN-US style='font-size:18.0pt;font-family:"Palatino Linotype",serif'>jinjian.wu@mail.xidian.edu.cn</span></p>

<p class=MsoNormal style='text-indent:72.0pt;line-height:28.0pt'><span
lang=EN-US style='font-size:18.0pt;font-family:"Palatino Linotype",serif'>Guangming
Shi, Email: gmshi@xidian.edu.cn</span></p>

<p class=MsoNormal style='line-height:12.0pt'><span lang=EN-US
style='font-size:12.0pt;font-family:"Times New Roman",serif;color:#0070C0'>&nbsp;</span></p>

<p class=MsoNormal style='line-height:12.0pt'><span lang=EN-US
style='font-size:12.0pt;font-family:"Times New Roman",serif;color:#0070C0'>&nbsp;</span></p>

<p class=MsoNormal style='line-height:12.0pt'><span lang=EN-US
style='font-size:12.0pt;font-family:"Times New Roman",serif;color:#0070C0'>&nbsp;</span></p>

<p class=MsoNormal style='line-height:12.0pt'><span lang=EN-US
style='font-size:12.0pt;font-family:"Times New Roman",serif;color:#0070C0'>&nbsp;</span></p>

<p class=MsoNormal style='line-height:12.0pt'><span lang=EN-US
style='font-size:12.0pt;font-family:"Times New Roman",serif;color:#0070C0'>&nbsp;</span></p>

<p class=MsoNormal style='line-height:12.0pt'><span lang=EN-US
style='font-size:12.0pt;font-family:"Times New Roman",serif;color:#0070C0'>&nbsp;</span></p>

<p class=MsoNormal style='line-height:12.0pt'><span lang=EN-US
style='font-size:12.0pt;font-family:"Times New Roman",serif;color:#0070C0'>&nbsp;</span></p>

</div>

</body>

</html>
