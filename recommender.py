    import time
    import re
    import numpy as np
    import scipy
    import pandas as pd
    import math
    import random
    import sklearn
    from nltk.corpus import stopwords
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.sparse.linalg import svds
    import matplotlib.pyplot as plt
    from math import sqrt
    from sklearn.metrics import mean_squared_error

t0 = time.clock()

answer_question_path = 'C:\Users\Nick Liu\Desktop\keen3.csv'

answer_question_df = pd.read_csv(answer_question_path)

answer_question_df = answer_question_df[answer_question_df.pageview > 100]

# answer_question_df = answer_question_df.head(20000)

answer_question_subset_df = answer_question_df[['request.siteUUID', 'questionUUID', 'engagement_rate']]



# Metrics
#=================================================================================================================================================================

####### Mean Square Error for Collaborative Filtering ########
n_full_site = len(set(answer_question_subset_df['request.siteUUID']))
n_full_question = len(set(answer_question_subset_df['questionUUID']))


def match_dic():
	siteUUID_match = {}
	siteUUID_index = 0
	for siteUUID in set(answer_question_subset_df['request.siteUUID']):
		siteUUID_match.update({siteUUID: siteUUID_index})
		siteUUID_index += 1

	questionUUID_match = {}
	questionUUID_index = 0
	for questionUUID in set(answer_question_subset_df['questionUUID']):
		questionUUID_match.update({questionUUID: questionUUID_index})
		questionUUID_index += 1

	return [siteUUID_match, questionUUID_match]

siteUUID_match = match_dic()[0]
questionUUID_match = match_dic()[1]


ans_que_train_df, ans_que_test_df = train_test_split(answer_question_subset_df, test_size = 0.30)

interactions_full_indexed_df = answer_question_subset_df.set_index('request.siteUUID')
interactions_train_indexed_df = ans_que_train_df.set_index('request.siteUUID')
interactions_test_indexed_df = ans_que_test_df.set_index('request.siteUUID')


train_data_matrix = np.zeros((n_full_site, n_full_question))
for line in ans_que_train_df.itertuples():
	train_data_matrix[siteUUID_match[line[1]], questionUUID_match[line[2]]] = line[3]

test_data_matrix = np.zeros((n_full_site, n_full_question))
for line in ans_que_test_df.itertuples():
	test_data_matrix[siteUUID_match[line[1]], questionUUID_match[line[2]]] = line[3]


def rmse(prediction, ground_truth):
	prediction = prediction[ground_truth.nonzero()].flatten()
	ground_truth = ground_truth[ground_truth.nonzero()].flatten()
	return sqrt(mean_squared_error(prediction, ground_truth))

######### Mean Average Precision ###########
interactions_train_df, interactions_test_df = train_test_split(answer_question_subset_df,
												test_size = 0.2,
												random_state = 42)

# print ('# interactions on Train set: %d' % len(interactions_train_df))
# print ('# interactions on Test set: %d' % len(interactions_test_df))

interactions_full_indexed_df = answer_question_subset_df.set_index('request.siteUUID')
interactions_train_indexed_df = interactions_train_df.set_index('request.siteUUID')
interactions_test_indexed_df = interactions_test_df.set_index('request.siteUUID')

# print ('interactions_test_df')
# print (interactions_test_df)

# print ('interactions_test_indexed_df')
# print (interactions_test_indexed_df)


def get_questions_interacted(site_id, interactions_indexed_df):
	try:
		interacted_questions = interactions_indexed_df.loc[site_id]['questionUUID']
	except:
		print (site_id)
		return 
	return set(interacted_questions if type(interacted_questions) == pd.Series else [interacted_questions])

# interacted_questions = get_questions_interacted(, interactions_full_indexed_df)
# print (interacted_questions)
# print (interactions_full_indexed_df)

EVAL_RANDOM_SAMPLE_NON_INTERACTED_QUESTIONS = 100

class ModelEvaluator:

	def get_not_interacted_questions_sample(self, site_id, sample_size, seed = 42):
		interacted_questions = get_questions_interacted(site_id, interactions_full_indexed_df)
		all_questions = set(answer_question_subset_df['questionUUID'])
		non_interacted_questions = all_questions - interacted_questions

		random.seed(seed)
		non_interacted_questions_sample = random.sample(non_interacted_questions, sample_size)
		return set(non_interacted_questions_sample)

	def _verify_hit_top_n(self, question_id, recommended_questions, topn):
		try:
			index = next(i for i, c in enumerate(recommended_questions) if c == question_id)
		except:
			index = -1
		hit = int(index in range(0, topn))
		return hit, index

	def evaluate_model_for_site(self, model, site_id):
		interacted_values_testset = interactions_test_indexed_df.loc[site_id]
		if type(interacted_values_testset['questionUUID']) == pd.Series:
			site_interacted_questions_testset = set(interacted_values_testset['questionUUID'])
		else:
			site_interacted_questions_testset = set([interacted_values_testset['questionUUID']])
		interacted_questions_count_testset = len(site_interacted_questions_testset)

		site_recs_df = model.recommend_questions(site_id,
												questions_to_ignore = get_questions_interacted(site_id,
																					   interactions_train_indexed_df),
																					   topn = 10000000000)
		hits_at_5_count = 0
		hits_at_10_count = 0

		for question_id in site_interacted_questions_testset:
			non_interacted_questions_sample = self.get_not_interacted_questions_sample(site_id, 
																			   sample_size = EVAL_RANDOM_SAMPLE_NON_INTERACTED_QUESTIONS,
																			   seed = 42)

		questions_to_filter_recs = non_interacted_questions_sample.union(set([question_id]))

		try:
			valid_recs_df = site_recs_df[site_recs_df['questionUUID'].isin(questions_to_filter_recs)]


			valid_recs = valid_recs_df['questionUUID'].values

			hits_at_5, index_at_5 = self._verify_hit_top_n(question_id, valid_recs, 5)
			hits_at_5_count += hits_at_5
			hits_at_10, index_at_10 = self._verify_hit_top_n(question_id, valid_recs, 10)
			hits_at_10_count += hits_at_10

			recall_at_5 = hits_at_5_count / float(interacted_questions_count_testset)
			recall_at_10 = hits_at_10_count / float(interacted_questions_count_testset)

			site_metrics = {'hits@5_count': hits_at_5_count,
							'hits@10_count': hits_at_10_count,
							'interacted_count': interacted_questions_count_testset,
							'recall@5': recall_at_5,
							'recall@10': recall_at_10}
			return site_metrics
		except:
			return

	def evaluate_model(self, model):
		sites_metrics = []
		# print (interactions_test_indexed_df.index.unique())
		for idx, site_id in enumerate(list(interactions_test_indexed_df.index.unique())):
			site_metrics = self.evaluate_model_for_site(model, site_id)
			if site_metrics:
				site_metrics['_site_id'] = site_id
				sites_metrics.append(site_metrics)
		print ('%d sites processed' % idx)

		try:
			detailed_results_df = pd.DataFrame(sites_metrics) \
								.sort_values('interacted_count', ascending = False)

			global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
			global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())

			global_metrics = {'modelName': model.get_model_name(),
							  'recall@5': global_recall_at_5,
							  'recall@10': global_recall_at_10}
			return global_metrics, detailed_results_df
		except:
			return

model_evaluator = ModelEvaluator()

# Content-based Algorithm
#================================================================================================================================================================

stopwords_list = stopwords.words('english') + stopwords.words('spanish')
# print (stopwords_list)
vectorizer = TfidfVectorizer(analyzer = 'word',
							ngram_range = (1, 3),
							min_df = 0.003,
							max_df = 0.8,
							max_features = 5000,
							stop_words = stopwords_list)

question_ids = answer_question_df['questionUUID'].tolist()


# Spanish Reading issue needs to be fixed
question_list = []
for question in answer_question_df['question'].tolist():
	try:
		question_list.append(str(question).encode('utf-8').strip())
	except:
		# try:
		# 	# print(re.sub(r'[^\w]', ' ', str(question)))
		question_list.append(re.sub(r'[^\w]', ' ', str(question)).encode('utf-8').strip())
		# except:
		# 	print(re.sub(r'[^\w]', ' ', str(question)))
		# 	pass
		# 	# print (question)


tfidf_matrix = vectorizer.fit_transform(question_list)
# print (dict(zip(vectorizer.get_feature_names(), vectorizer.idf_)))
tfidf_feature_names = vectorizer.get_feature_names()


def get_question_profile(question_id):
	idx = question_ids.index(question_id)
	question_profile = tfidf_matrix[idx]
	return question_profile

def get_question_profiles(ids):
	question_profiles_list = []
	if type(ids) != str:
		question_profiles_list = [get_question_profile(x) for x in ids]
	else:
		question_profiles_list.append(get_question_profile(ids))
	question_profiles = scipy.sparse.vstack(question_profiles_list)
	return question_profiles

def build_sites_profile(site_id, interactions_indexed_df):
	interactions_site_df = interactions_indexed_df.loc[site_id]
	site_question_profiles = get_question_profiles(interactions_site_df['questionUUID'])
	site_question_strengths = np.array(interactions_site_df['engagement_rate']).reshape(-1,1)
	rows, cols = site_question_profiles.shape
	if rows > 1:
		site_question_strengths_weighted_avg = np.sum(site_question_profiles.multiply(site_question_strengths), axis=0) / np.sum(site_question_strengths)
	else:
		site_question_strengths_weighted_avg = site_question_profiles 
	site_profile_norm = sklearn.preprocessing.normalize(site_question_strengths_weighted_avg)
	return site_profile_norm

def build_sites_profiles(): 
	interactions_indexed_df = answer_question_subset_df.set_index('request.siteUUID')
	site_profiles = {}
	for site_id in interactions_indexed_df.index.unique():
		site_profiles[site_id] = build_sites_profile(site_id, interactions_indexed_df)
	return site_profiles

site_profiles = build_sites_profiles()
	
myprofile = site_profiles['91a526a7-de27-41a7-b1f5-fd7e5bb416f8']

try:
	print (pd.DataFrame(sorted(zip(tfidf_feature_names, myprofile.flatten().tolist()), key=lambda x: -x[1])[:20],
			 columns = ['token', 'relevance']))
except:
	pass

class ContentBasedRecommender:

	MODEL_NAME = 'Content-Based'

	def __init__(self, questions_df = None):
		self.question_ids = question_ids
		self.questions_df = questions_df

	def get_model_name(self):
		return self.MODEL_NAME

	def _get_similar_questions_to_site_profile(self, site_id, topn = 1000):
		cosine_similarities = cosine_similarity(site_profiles[site_id], tfidf_matrix)
		similar_indices = cosine_similarities.argsort().flatten()[-topn:]
		similar_questions = sorted([(question_ids[i], cosine_similarities[0, i]) for i in similar_indices], key = lambda x: -x[1])
		return similar_questions

	def recommend_questions(self, site_id, questions_to_ignore = [], topn = 10, verbose = False):
		# if questions_to_ignore:
		similar_questions = self._get_similar_questions_to_site_profile(site_id)
		similar_questions_filtered = list(filter(lambda x: x[0] not in questions_to_ignore, similar_questions))

		recommendations_df = pd.DataFrame(similar_questions_filtered, columns = ['questionUUID', 'recEngagement']).head(topn)

		if verbose:
			if self.questions_df is None:
				raise Exception('"questions_df" is required in verbose mode')

			recommendations_df = recommendations_df.merge(self.questions_df, how = 'left',
															  left_on = 'questionUUID',
															  right_on = 'questionUUID') [['recEngagement', 'questionUUID', 'question']]

		return recommendations_df
		# return

content_based_recommender_model = ContentBasedRecommender(answer_question_df)

print ('content_based_recommender_model')
print (content_based_recommender_model.recommend_questions('91a526a7-de27-41a7-b1f5-fd7e5bb416f8', topn=20, verbose=True))
print (np.unique(content_based_recommender_model.recommend_questions('91a526a7-de27-41a7-b1f5-fd7e5bb416f8', topn=20, verbose=True)[['questionUUID']]))




# Collabarative Filtering
#===========================================================================================================================================================

sites_questions_full_pivot_matrix_df = answer_question_subset_df.pivot(index = 'request.siteUUID',
														columns = 'questionUUID',
														values = 'engagement_rate').fillna(0)

site_full_ids = list(sites_questions_full_pivot_matrix_df.index)
# question_full_ids = list(sites_questions_full_pivot_matrix_df.columns)

NUMBER_OF_FACTORS_MF = 20


U_train, sigma_train, Vt_train = svds(train_data_matrix, k = NUMBER_OF_FACTORS_MF)

sigma_train = np.diag(sigma_train)


sites_predicted_questions_train_answered = np.dot(np.dot(U_train, sigma_train), Vt_train)

sparsity = round(1.0 - len(answer_question_df)/float(n_full_site*n_full_question), 3)

# print ('The sparsity level of site_Question_Answer is ' + str(sparsity*100) + '%')

# print ('site Based CF MSE: ' + str(rmse(sites_predicted_questions_train_answered, test_data_matrix)/10000000))


# sites_ids = list(sites_questions_pivot_matrix_df.index)

cf_preds_df= pd.DataFrame(sites_predicted_questions_train_answered, columns = sites_questions_full_pivot_matrix_df.columns, index = site_full_ids).transpose()


class CFRecommender:

	MODEL_NAME = 'Collaborative Filtering'

	def __init__(self, cf_predictions_df, questions_df = None):
		self.cf_predictions_df = cf_predictions_df
		self.questions_df = questions_df

	def get_model_name(self):
		return self.MODEL_NAME

	def recommend_questions(self, site_id, questions_to_ignore = [], topn = 10, verbose = False):

		sorted_site_predictions = self.cf_predictions_df[site_id].sort_values(ascending = False) \
									.reset_index().rename(columns = {site_id: 'recEngagement'})

		try:
			recommendations_df = sorted_site_predictions[~sorted_site_predictions['questionUUID'].isin(questions_to_ignore)] \
								.sort_values('recEngagement', ascending = False) \
								.head(topn)

			if verbose:
				if self.questions_df is None:
					raise Exception ('"questions_df" is required in verbose mode')

				recommendations_df = recommendations_df.merge(self.questions_df, how = 'left',
															left_on = 'questionUUID',
															right_on = 'questionUUID') [['recEngagement', 'questionUUID', 'question']]

			return recommendations_df
		except:
			return



cf_recommender_model = CFRecommender(cf_preds_df, answer_question_df)

# print ('Collaborative Filtering')
# print (cf_recommender_model.recommend_questions('be93700b-57aa-4ac1-9e4f-157a7b847385', topn=20, verbose=True))


# Hybrid
#======================================================================================================================================================

class HybridRecommender:

	MODEL_NAME = 'Hybrid'

	def __init__(self, cb_rec_model, cf_rec_model, questions_df):
		self.cb_rec_model = cb_rec_model
		self.cf_rec_model = cf_rec_model
		self.questions_df = questions_df

	def get_model_name(self):
		return self.MODEL_NAME

	def recommend_questions(self, site_id, questions_to_ignore=[], topn = 10, verbose = False):
		try:
			cb_recs_df = self.cb_rec_model.recommend_questions(site_id, questions_to_ignore = questions_to_ignore, verbose = verbose, topn = 1000) \
					.rename(columns = {'recEngagement': 'recStrengthCB'})
			# print (cb_recs_df)

			cf_recs_df = self.cf_rec_model.recommend_questions(site_id, questions_to_ignore = questions_to_ignore, verbose = verbose, topn = 1000) \
						.rename(columns = {'recEngagement': 'recStrengthCF'})

			recs_df = cb_recs_df.merge(cf_recs_df, 
										how = 'inner',
										left_on = 'questionUUID',
										right_on = 'questionUUID')

			recs_df['recStrengthHybrid'] = recs_df['recStrengthCB'] + recs_df['recStrengthCF']

			recommendations_df = recs_df.sort_values('recStrengthHybrid', ascending = False).head(topn)

			if verbose:
				if self.questions_df is None:
					raise Exception('"questions_df" is required in verbose mode')

				recommendations_df = recommendations_df.merge(self.questions_df, how = 'left',
															  left_on = 'questionUUID',
															  right_on = 'questionUUID')[['recStrengthHybrid', 'questionUUID', 'question']]
			return recommendations_df
		except:
			return 

hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, answer_question_df)


# def inspect_interactions(site_id, test_set = True):
# 	if test_set:
# 		interactions_df = interactions_test_indexed_df
# 	else:
# 		interactions_df = interactions_train_indexed_df

# 	return interactions_df.loc[site_id].merge(answer_question_df, how = 'left',
# 															left_on = 'questionUUID',
# 															right_on = 'questionUUID') \
# 						  .sort_values('engagement_rate', ascending = False)[['questionUUID', 'question', 'site']]

# print ('be93700b-57aa-4ac1-9e4f-157a7b847385')
# # print ('actually interacted')
# # print (inspect_interactions('be93700b-57aa-4ac1-9e4f-157a7b847385', test_set = False).head(20))

# print ('Hybrid')
# print (hybrid_recommender_model.recommend_questions('91a526a7-de27-41a7-b1f5-fd7e5bb416f8', topn=100, verbose=True))
# print (np.unique(hybrid_recommender_model.recommend_questions('91a526a7-de27-41a7-b1f5-fd7e5bb416f8', topn=100, verbose=True)[['question']]))


# print ('Evaluating cb model...')
# cb_global_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(content_based_recommender_model)
# print ('\nGlobal metrics:\n%s' % cb_global_metrics)
# print (cb_detailed_results_df.head(10))

t1 = time.clock()
time_spend = t1 - t0
print (str(time_spend) + ' s')