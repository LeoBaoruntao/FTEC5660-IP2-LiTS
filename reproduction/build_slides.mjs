import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const runtime='C:/Users/babyt/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=runtime+'/node/node_modules';
const skill='C:/Users/babyt/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const root=path.resolve('.');
const {Presentation,PresentationFile}=await import(pathToFileURL(runtime+'/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs'));
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs'));
const content=JSON.parse(await fs.readFile('submission/slides_content.json','utf8'));
const pres=Presentation.create({slideSize:{width:1280,height:720}});
await fs.mkdir('.presentation-build',{recursive:true});
function text(slide,txt,x,y,w,h,size,color='#183249',bold=false){
  const s=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  s.text=txt;s.text.style={typeface:'Arial',fontSize:size,color,bold};return s;
}
for(let i=0;i<content.length;i++){
 const [title,body,notes]=content[i];const slide=pres.slides.add();slide.background.fill='#F7F8FA';
 text(slide,title,70,65,1140,150,i===0?58:49,'#17364F',true);
 text(slide,body,74,255,1120,330,32,'#263E50');
 text(slide,'FTEC5660 · Individual Project 2',74,650,1000,35,17,'#647587');
 text(slide,String(i+1),1160,650,50,35,17,'#647587');
 slide.speakerNotes.text=notes+'\nSources: https://github.com/xinzhel/lits-llm/tree/25e94d862cb4514c22d9a80a635e392461769df7 ; reproduction/analysis/summary.json; reproduction/DEBUG_DIARY.md.';
 const blob=await pres.export({slide,format:'png',scale:1});
 await fs.writeFile(`.presentation-build/slide-${i+1}.png`,new Uint8Array(await blob.arrayBuffer()));
}
const candidate=path.join(root,'.presentation-build/draft.pptx');
await (await PresentationFile.exportPptx(pres)).save(candidate);
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:path.join(root,'submission/Presentation.pptx'),
 pythonExecutable:runtime+'/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],
 explicitTotalSlideCount:8,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'design',families:['Arial']},
 verifyArtifactToolImport:true,receiptPath:path.join(root,'.presentation-build/validation.json')});
console.log('Eight slides exported and finalized.');
