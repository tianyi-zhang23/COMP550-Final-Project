from fastai.text.all import *
import torchtext
from args import args_parser
from utils import *

def main():
    random.seed(10)
    args = args_parser
    
    train_df, test_df = get_dataset(args.dataset, args.size)
    augment = []
    if args.randswap:
        swapped = train_df.copy()
        swapped.apply(lambda row: random_swap(row['text'],args.p))
        augment.append(swapped)
    if args.randdel:
        deleted = train_df.copy()
        deleted.apply(lambda row: random_deletion(row['text'], args.p))
        augment.append(deleted)
    if args.synrep:
        synonym_rep = train_df.copy()
        synonym_rep.apply(lambda row: random_deletion(row['text'], args.p))
        augment.append(synonym_rep)
    if args.randin:
        randin = train_df.copy()
        randin.apply(lambda row: random_deletion(row['text'], args.p))
        augment.append(randin)

    for aug in augment:
        train_df.append(augment)
    train_df.drop_duplicates()
    
    
    train_lm = TextDataLoaders.from_df(train_df, text_col='text', is_lm=True)
    train_lm.show_batch()
    
    learn = language_model_learner(train_lm, AWD_LSTM, metrics=[accuracy, Perplexity()], wd=0.1).to_fp16()
    
    learn.fit_one_cycle(1, 1e-2)
    learn.unfreeze()
    learn.fit_one_cycle(5, 1e-3)
    
    learn.save_encoder('finetuned')
    
    
    train_class = TextDataLoaders.from_df(train_df, text_col='text', label_col='label' ,text_vocab=train_lm.vocab)
    train_class.show_batch()
    
    learn = text_classifier_learner(train_class, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
    learn = learn.load_encoder('finetuned')
    
    learn.freeze_to(-2)
    learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2))
    
    learn.freeze_to(-3)
    learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))
    
    test_dl = learn.dls.test_dl(test_df, with_label= True )
    acc = learn.validate(dl=test_dl)[1]
    print(acc)
    return


if __name__ == "__main__":
    main()
