import pandas as pd
import json
import spacy
import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str) # a json file
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--summary_type', type=str) # layman or expert
    args = parser.parse_args()

    with open(args.input_dir, 'r', encoding='utf-8') as f:
        load_dict = json.load(f)

    print(type(load_dict))
    if args.summary_type == 'layman':
        
        df = pd.DataFrame([[item['layman_summary'], item['document'], 0.0] for item in load_dict], columns=['summary', 'source', 'faithfulness'])
    elif args.summary_type == 'expert':
        df = pd.DataFrame([[item['expert'], item['document'], 0.0] for item in load_dict], columns=['summary', 'source', 'faithfulness']) 
    df.to_csv("data.csv")

    def prepare_dataset(file_path):
            print("loading input file...")
            df = pd.read_csv(file_path)

            spacymodel = 'en_core_web_lg'
            print(f'Loading Spacy model {spacymodel}...')
            nlp = spacy.load(spacymodel)

            output = [{
                "source": source,
                "summary": summary,
                "faithfulness": faithfulness,
                "summary_sentences": [x.text for x in nlp(summary).sents],
                "source_sentences": [x.text for x in nlp(source).sents]
            } for summary, source, faithfulness in zip(df["summary"].tolist(), df["source"].tolist(), df["faithfulness"].tolist())]

            # Save output
            out_file = "prepared_" + args.dataset_name + '.json'
            with open(out_file, 'w', encoding='utf-8') as file:
                json.dump(output, file)

    prepare_dataset("elife_layman.csv")

if __name__ == '__main__':
    main()