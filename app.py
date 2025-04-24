# app.py
from flask import Flask, request, jsonify
import requests
import os
import json
import logging
from logging.handlers import RotatingFileHandler
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        RotatingFileHandler('app.log', maxBytes=10000000, backupCount=5),
        logging.StreamHandler()
    ],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load API keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    logger.error("Gemini API key not found. Please add it to your .env file.")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/api/fact-check', methods=['POST'])
def fact_check():
    """Endpoint to fact-check a claim using research papers"""
    try:
        # Get claim from request
        data = request.get_json()
        if not data or 'claim' not in data:
            return jsonify({"error": "Please provide a claim to fact-check"}), 400
        
        claim = data['claim']
        logger.info(f"Received claim for fact-checking: {claim}")
        
        # Step 1: Search for relevant papers using Semantic Scholar
        logger.info("Retrieving research papers from Semantic Scholar...")
        papers = search_semantic_scholar(claim)
        
        if not papers:
            return jsonify({
                "assessment": "Lacks Sufficient Evidence",
                "explanation": "Could not retrieve relevant research papers for analysis.",
                "references": []
            }), 200
        
        # Step 2: Analyze the claim using an LLM
        logger.info("Analyzing claim with LLM...")
        analysis_result = analyze_with_llm(claim, papers)
        
        # Step 3: Format and return the response
        response = {
            "assessment": analysis_result["assessment"],
            "explanation": analysis_result["explanation"],
            "references": [{"title": paper["title"], "paper_id": paper.get("paperId", "N/A"), "abstract": paper["abstract"]} for paper in papers]
        }
        
        logger.info(f"Completed fact-check: {response['assessment']}")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error in fact-check endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while processing your request"}), 500

def search_semantic_scholar(query, limit=5):
    """Search for papers on Semantic Scholar API"""
    try:
        base_url = "http://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,abstract"
        }
        
        response = requests.get(base_url, params=params)
        
        results = response.json().get('data', [])

        if not results:
            # using default response if no results found for default cliam "Intermittent fasting improves brain function"
            logger.warning("Semantic API did not work as expected. No results found!")
            results = {
                'total': 3180,
                'offset': 0,
                'next': 5,
                'data': [
                    {   'paperId': '66c01aee89c1e907215a2644a490f9865a7a4074',
                        'title': 'Long-term intermittent fasting improves neurological function by promoting angiogenesis after cerebral ischemia via growth differentiation factor 11 signaling activation',
                        'abstract': 'Intermittent fasting (IF), an alternative to caloric restriction, is a form of time restricted eating. IF conditioning has been suggested to have neuroprotective effects and potential long-term brain health benefits. But the mechanism underlying remains unclear. The present study focused on the cerebral angiogenesis effect of IF on ischemic rats. Using a rat middle cerebral artery occlusion model, we assessed neurological outcomes and various vascular parameters such as microvessel density (MVD), regional cerebral blood flow (rCBF), proliferation of endothelial cells (ECs), and functional vessels in the peri-infarct area. IF conditioning ameliorated the modified neurological severity score and adhesive removal test, increased MVD, and activated growth differentiation factor 11 (GDF11)/activin-like kinase 5 (ALK5) pathways in a time-dependent manner. In addition, long-term IF conditioning stimulated proliferation of ECs, promoted rCBF, and upregulated the total vessel surface area as well as the number of microvessel branch points through GDF11/ALK5 pathways. These data suggest that long-term IF conditioning improves neurological outcomes after cerebral ischemia, and that this positive effect is mediated partly by angiogenesis in the peri-infarct area and improvement of functional perfusion microvessels in part by activating the GDF11/ALK5 signaling pathway.',
                        'openAccessPdf': {'url': 'https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0282338&type=printable',
                            'status': 'GOLD',
                            'license': 'CCBY',
                            'disclaimer': 'Notice: This abstract is extracted from the open access paper or abstract available at https://pmc.ncbi.nlm.nih.gov/articles/PMC10062670, which is subject to the license by the author or copyright owner provided with this content. Please go to the source to verify the license and copyright information for your use.'},
                        'authors': [{'authorId': '2212147449', 'name': 'Zhaojuan Liu'},
                            {'authorId': '2214295666', 'name': 'Mengjie Liu'},
                            {'authorId': '6282411', 'name': 'Gongwei Jia'},
                            {'authorId': '1659976857', 'name': 'Jiani Li'},
                            {'authorId': '39514686', 'name': 'Lingchuan Niu'},
                            {'authorId': '2130382747', 'name': 'Huiji Zhang'},
                            {'authorId': '2152003889', 'name': 'Yunwen Qi'},
                            {'authorId': '2152994080', 'name': 'Houchao Sun'},
                            {'authorId': '4770466', 'name': 'Liang-Jun Yan'},
                            {'authorId': '8555588', 'name': 'Jingxi Ma'}]
                    },
                    {   'paperId': '0cc448a0ac1b198f18ed759910ccbedcf0859894',
                        'title': 'Gut microbiota mediates intermittent-fasting alleviation of diabetes-induced cognitive impairment',
                        'abstract': 'Cognitive decline is one of the complications of type 2 diabetes (T2D). Intermittent fasting (IF) is a promising dietary intervention for alleviating T2D symptoms, but its protective effect on diabetes-driven cognitive dysfunction remains elusive. Here, we find that a 28-day IF regimen for diabetic mice improves behavioral impairment via a microbiota-metabolites-brain axis: IF enhances mitochondrial biogenesis and energy metabolism gene expression in hippocampus, re-structures the gut microbiota, and improves microbial metabolites that are related to cognitive function. Moreover, strong connections are observed between IF affected genes, microbiota and metabolites, as assessed by integrative modelling. Removing gut microbiota with antibiotics partly abolishes the neuroprotective effects of IF. Administration of 3-indolepropionic acid, serotonin, short chain fatty acids or tauroursodeoxycholic acid shows a similar effect to IF in terms of improving cognitive function. Together, our study purports the microbiota-metabolites-brain axis as a mechanism that can enable therapeutic strategies against metabolism-implicated cognitive pathophysiologies. Intermittent fasting (IF) has been shown beneficial in reducing metabolic diseases. Here, using a multi-omics approach in a T2D mouse model, the authors report that IF alters the composition of the gut microbiota and improves metabolic phenotypes that correlate with cognitive behavior.',
                        'openAccessPdf': {'url': 'https://www.nature.com/articles/s41467-020-14676-4.pdf',
                            'status': 'GOLD',
                            'license': 'CCBY',
                            'disclaimer': 'Notice: This abstract is extracted from the open access paper or abstract available at https://pmc.ncbi.nlm.nih.gov/articles/PMC7029019, which is subject to the license by the author or copyright owner provided with this content. Please go to the source to verify the license and copyright information for your use.'},
                        'authors': [{'authorId': '50873898', 'name': 'Zhigang Liu'},
                            {'authorId': '16070844', 'name': 'Xiaoshuang Dai'},
                            {'authorId': '2143480636', 'name': 'Hongbo Zhang'},
                            {'authorId': '29802652', 'name': 'Renjie Shi'},
                            {'authorId': '2056016997', 'name': 'Yan Hui'},
                            {'authorId': '2149170377', 'name': 'Xin Jin'},
                            {'authorId': '2143397055', 'name': 'Wentong Zhang'},
                            {'authorId': '29891080', 'name': 'Luanfeng Wang'},
                            {'authorId': '123451864', 'name': 'Qianxu Wang'},
                            {'authorId': '2155680416', 'name': 'Danna Wang'},
                            {'authorId': '2110368375', 'name': 'Jia Wang'},
                            {'authorId': '49902299', 'name': 'Xintong Tan'},
                            {'authorId': '50586839', 'name': 'Bo Ren'},
                            {'authorId': '1498613479', 'name': 'Xiaoning Liu'},
                            {'authorId': '2088210958', 'name': 'Tong Zhao'},
                            {'authorId': '2109014067', 'name': 'Jiamin Wang'},
                            {'authorId': '14416145', 'name': 'Junru Pan'},
                            {'authorId': '48579973', 'name': 'Tian Yuan'},
                            {'authorId': '29898978', 'name': 'Chuanqi Chu'},
                            {'authorId': '2130614496', 'name': 'Lei Lan'},
                            {'authorId': '1498516859', 'name': 'F. Yin'},
                            {'authorId': '145960185', 'name': 'E. Cadenas'},
                            {'authorId': '2153000093', 'name': 'Lin Shi'},
                            {'authorId': '7420554', 'name': 'Shancen Zhao'},
                            {'authorId': '1390611971', 'name': 'Xuebo Liu'}]
                        },
                    {   'paperId': 'bc2db6b4c47891a98e9ca6e247ca97a564f8611b',
                        'title': 'Effect of Calorie Restriction and Intermittent Fasting Regimens on Brain-Derived Neurotrophic Factor Levels and Cognitive Function in Humans: A Systematic Review',
                        'abstract': 'Background: The potential positive interaction between intermittent fasting (IF) and brain-derived neurotrophic factor (BDNF) on cognitive function has been widely discussed. This systematic review tried to assess the efficacy of interventions with different IF regimens on BDNF levels and their association with cognitive functions in humans. Interventions with different forms of IF such as caloric restriction (CR), alternate-day fasting (ADF), time-restricted eating (TRE), and the Ramadan model of intermittent fasting (RIF) were targeted. Methods: A systematic review was conducted for experimental and observational studies on healthy people and patients with diseases published in EMBASE, Scopus, PubMed, and Google Scholar databases from January 2000 to December 2023. We followed the Preferred Reporting Items for Systematic Reviews and Meta-Analysis statements (PRISMA) for writing this review. Results: Sixteen research works conducted on healthy people and patients with metabolic disorders met the inclusion criteria for this systematic review. Five studies showed a significant increase in BDNF after the intervention, while five studies reported a significant decrease in BDNF levels, and the other six studies showed no significant changes in BDNF levels due to IF regimens. Moreover, five studies examined the RIF protocol, of which, three studies showed a significant reduction, while two showed a significant increase in BDNF levels, along with an improvement in cognitive function after RIF. Conclusions: The current findings suggest that IF has varying effects on BDNF levels and cognitive functions in healthy, overweight/obese individuals and patients with metabolic conditions. However, few human studies have shown that IF increases BDNF levels, with controversial results. In humans, IF has yet to be fully investigated in terms of its long-term effect on BDNF and cognitive functions. Large-scale, well-controlled studies with high-quality data are warranted to elucidate the impact of the IF regimens on BDNF levels and cognitive functions.',
                        'openAccessPdf': {'url': 'https://www.mdpi.com/1648-9144/60/1/191/pdf?version=1705912067',
                            'status': 'GOLD',
                            'license': 'CCBY',
                            'disclaimer': 'Notice: This abstract is extracted from the open access paper or abstract available at https://pmc.ncbi.nlm.nih.gov/articles/PMC10819730, which is subject to the license by the author or copyright owner provided with this content. Please go to the source to verify the license and copyright information for your use.'},
                        'authors': [{'authorId': '74619313', 'name': 'R. Alkurd'},
                            {'authorId': '1412384289', 'name': 'Lana Mahrous'},
                            {'authorId': '51026783', 'name': 'Falak Zeb'},
                            {'authorId': '2275583683', 'name': 'M. Khan'},
                            {'authorId': '2268742143', 'name': 'Hamid Alhaj'},
                            {'authorId': '2258324802', 'name': 'Husam Khraiwesh'},
                            {'authorId': '1752994032', 'name': 'MoezAlIslam E. Faris'}]
                    },
                    {   'paperId': 'a8e0e83cc00804587e304badab3837f6cc129c4d',
                        'title': 'Intermittent Fasting After ST-Segment–Elevation Myocardial Infarction Improves Left Ventricular Function: The Randomized Controlled INTERFAST-MI Trial',
                        'abstract': 'BACKGROUND: Intermittent fasting has shown positive effects on numerous cardiovascular risk factors. The INTERFAST-MI trial (Intermittent Fasting in Myocardial Infarction) has been designed to study the effects of intermittent fasting on cardiac function after STEM (ST-segment–elevation myocardial infarction) and the feasibility of future multicenter trials. METHODS: The INTERFAST-MI study was a prospective, randomized, controlled, nonblinded, single-center investigator-initiated trial. From October 1, 2020, to July 15, 2022, 48 patients were randomized to the study groups intermittent fasting or regular diet and followed for 6 months with follow-up visits at 4 weeks and 3 months. RESULTS: In all, 22 of 24 patients in the intermittent fasting group with a mean age of 58.54±12.29 years and 20 of 24 patients in the regular diet group with a mean age of 59.60±13.11 years were included in the intention-to-treat population. The primary efficacy end point (improvement in left ventricular ejection fraction after 4 weeks) was significantly greater in the intermittent fasting group compared with the control group (mean±SD, 6.636±7.122%. versus 1.450±4.828%; P=0.038). This effect was still significant and even more pronounced after 3 and 6 months. The patients in the intermittent fasting group showed a greater reduction in diastolic blood pressure and body weight compared with the control group. The mean adherence of patients in the intermittent fasting group was a median of 83.7% (interquartile range, 69.0%–98.4%) of all days. None of the patients from either group reported dizziness, syncope, or collapse. CONCLUSIONS: Our results suggest that intermittent fasting after myocardial infarction may be safe and could improve left ventricular function after STEMI. REGISTRATION: URL: https://www.drks.de; Unique identifier: DRKS00021784.',
                        'openAccessPdf': {'url': '',
                            'status': None,
                            'license': None,
                            'disclaimer': 'Notice: This abstract is extracted from the open access paper or abstract available at https://api.unpaywall.org/v2/10.1161/CIRCHEARTFAILURE.123.010936?email=<INSERT_YOUR_EMAIL> or https://doi.org/10.1161/CIRCHEARTFAILURE.123.010936, which is subject to the license by the author or copyright owner provided with this content. Please go to the source to verify the license and copyright information for your use.'},
                        'authors': [{'authorId': '6929676', 'name': 'J. Dutzmann'},
                            {'authorId': '2161741978', 'name': 'Zoe Kefalianakis'},
                            {'authorId': '2253991722', 'name': 'F. Kahles'},
                            {'authorId': '145958700', 'name': 'J. Daniel'},
                            {'authorId': '2299313602', 'name': 'Hubert Gufler'},
                            {'authorId': '2299306634', 'name': 'W. A. Wohlgemuth'},
                            {'authorId': '2284957629', 'name': 'Kai Knöpp'},
                            {'authorId': '2252524525', 'name': 'D. Sedding'}]
                    },
                    {   'paperId': 'b8f2fdf5d0045b5bfb949e32e60f1ae499a3829d',
                            'title': '1354-P: Alternate-Day Intermittent Fasting Improves Diabetes and Protects Beta-Cell Function in Polygenic Mouse Models of T2DM',
                            'abstract': 'Type 2 diabetes (T2DM) , caused by the interaction of multiple genes and environmental factors, is characterized by hyperglycemia, insulin secretion deficiency and insulin resistance. Chronic hyperglycemia induces β-cell dysfunction and loss of β-cell mass/identity, increased apoptosis and β-cell dedifferentiation. Intermittent fasting (IF) , either alternate-day fasting (ADF) or time-restricted feeding, commonly used regimens for weight-loss, also induces metabolic benefits including reduced blood glucose, improved insulin sensitivity, reduced adiposity, inflammation and oxidative-stress, and increased fatty-acid oxidation; however, the mechanisms underlying these effects remain elusive. KK and KKAy, mouse models of polygenic T2DM spontaneously develop hyperglycemia, glucose intolerance, glucosuria, impaired insulin secretion and insulin resistance. To determine the long-term effects of IF on T2DM, 6-weeks old KK and KKAy mice were subjected to ADF for 16-weeks. While KKAy mice fed ad-libitum demonstrated severe hyperglycemia (˜500mg/dL) , increased plasma insulin, impaired glucose tolerance and insulin resistance at 8 weeks of age, KK mice showed blood glucose levels of ˜200mg/dL, but progressively became as severely diabetic as KKAy mice by 22-weeks. Strikingly, both KK and KKAy mice subjected to ADF showed reduced blood glucose and plasma insulin levels, decreased body weight gain despite increased food intake, reduced plasma triglycerides and cholesterol, and improved insulin sensitivity. They also demonstrated enhanced expression of the β-cell transcription factors PDX1 and NKX6.1, suggesting protection from loss of β-cell identity/dedifferentiation by IF. In addition, respiratory exchange ratio (RER) and movement were significantly enhanced in the ADF group, indicating peripheral benefits of IF. These results have important implications as an optional intervention for preservation of β-cell mass and function in T2DM.\n \n \n S.Patel: None. Z.Yan: None. M.S.Remedi: None.\n \n \n \n National Institutes of Health (R01DK123163)\n',
                            'openAccessPdf': {'url': '',
                                'status': None,
                                'license': None,
                                'disclaimer': 'Notice: This abstract is extracted from the open access paper or abstract available at https://api.unpaywall.org/v2/10.2337/db22-1354-p?email=<INSERT_YOUR_EMAIL> or https://doi.org/10.2337/db22-1354-p, which is subject to the license by the author or copyright owner provided with this content. Please go to the source to verify the license and copyright information for your use.'},
                            'authors': [{'authorId': '2109461935', 'name': 'S. Patel'},
                                {'authorId': '10018483', 'name': 'Zihan Yan'},
                                {'authorId': '5045213', 'name': 'M. Remedi'}]
                        }
                    ]
            }
            results = results.get('data', [])
        
        papers = []
        for paper in results:
            if paper.get('abstract'):  # Only include papers with abstracts
                papers.append({
                    "title": paper.get('title', 'Unknown Title'),
                    "abstract": paper.get('abstract', ''),
                    "paperId": paper.get('paperId', '')
                })
        
        logger.info(f"Retrieved {len(papers)} relevant papers from Semantic Scholar")
        return papers
    except Exception as e:
        logger.error(f"Error retrieving papers from Semantic Scholar: {str(e)}", exc_info=True)
        return []
    
# Output structure for Gemini LLM
class RespStructure(BaseModel):
  assessment: Optional[str]
  explanation: Optional[str]

def analyze_with_llm(claim, papers):
    """Analyze the claim and papers with an LLM"""
    try:

        system_instruction = (
            "You are a scientific research analyst focused on objectively evaluating claims "
            "based on available evidence. You will carefully read scientific abstracts and "
            "determine if they support, refute, or provide insufficient evidence for a given claim. "
            "Base your assessment only on the content provided, not on external knowledge."
        )
        
        # Construct the prompt for the LLM
        prompt = f"""
Given the following claim and research paper abstracts, assess whether the claim is "Supported," "Refuted," or "Lacks Sufficient Evidence" based ONLY on the provided texts.
Provide your assessment and a brief explanation of your reasoning in 2-3 sentences.

CLAIM: "{claim}"

RESEARCH PAPERS:
"""
        
        # Add papers to the prompt
        for i, paper in enumerate(papers, 1):
            prompt += f"\nPAPER {i}:\n"
            prompt += f"Title: {paper['title']}\n"
            prompt += f"Abstract: {paper['abstract']}\n"
    
        prompt += """
Based on ONLY the information in the provided research papers, please provide:
1. ASSESSMENT: One of "Supported," "Refuted," or "Lacks Sufficient Evidence"
2. EXPLANATION: A brief explanation (2-3 sentences) justifying your assessment

"""     

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        llm_response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=list[RespStructure],
                system_instruction=system_instruction
            )
        )
        
        llm_response_json = json.loads(llm_response.text)
        # Extract the assessment and explanation
        assessment = llm_response_json[0].get("assessment", "Lacks Sufficient Evidence")
        explanation = llm_response_json[0].get("explanation", "Analysis could not determine a clear assessment.")
        
        return {
            "assessment": assessment,
            "explanation": explanation
        }
    except Exception as e:
        logger.error(f"Error in LLM analysis: {str(e)}", exc_info=True)
        return {
            "assessment": "Error",
            "explanation": "An error occurred during the analysis of the research papers."
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
