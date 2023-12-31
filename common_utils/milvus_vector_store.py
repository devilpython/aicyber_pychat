import logging
from llama_index.vector_stores import MilvusVectorStore
from typing import Any, List, Optional
from llama_index.schema import TextNode
from llama_index.vector_stores.types import (
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import (
    metadata_dict_to_node,
)
from common_utils.redis_utils import RedisUtils
import config

logger = logging.getLogger(__name__)

MILVUS_ID_FIELD = "id"

def _to_milvus_filter(standard_filters: MetadataFilters) -> List[str]:
    """Translate standard metadata filters to Milvus specific spec."""
    filters = []
    for filter in standard_filters.filters:
        if isinstance(filter.value, str):
            filters.append(str(filter.key) + " == " + '"' + str(filter.value) + '"')
        else:
            filters.append(str(filter.key) + " == " + str(filter.value))
    return filters

class CUMilvusVectorStore(MilvusVectorStore):
    # 是否QA问答语料
    is_qa: bool = False

    # 向量距离
    radius: float = 0.5

    def __init__(
            self,
            is_qa: bool = False,
            uri: str = "http://localhost:19530",
            token: str = "",
            collection_name: str = "llamalection",
            dim: Optional[int] = None,
            radius: float = 0.5,
    ) -> None:
        MilvusVectorStore.__init__(self, uri, token, collection_name, dim=dim)
        self.is_qa = is_qa
        self.radius = radius

    def query_id(self, data_id: str) -> bool:
        return len(self.milvusclient.get(self.collection_name, [data_id], ['id'])) > 0

    def get_collection_name(self) -> str:
        return self.collection_name

    def query_ids(self, data_id_list: list) -> list:
        id_list = []

        window_num = int(len(data_id_list) / 200)
        mode_num = len(data_id_list) % 200
        for i in range(window_num):
            data_list = self.milvusclient.get(self.collection_name, data_id_list[i * 200:(i + 1) * 200], ['id'])
            # print('......query_ids data_id_set len:', len(set(data_id_list)))
            # print('......query_ids data_list len:', len(data_list))
            for data in data_list:
                id_list.append(data['id'])
        if mode_num > 0:
            data_list = self.milvusclient.get(self.collection_name, data_id_list[window_num * 200:len(data_id_list)], ['id'])
            # print('......query_ids data_id_set len:', len(set(data_id_list)))
            # print('......query_ids data_list len:', len(data_list))
            for data in data_list:
                id_list.append(data['id'])
        # print('......query_ids id_list:', id_list)
        return id_list

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes
            doc_ids (Optional[List[str]]): list of doc_ids to filter by
            node_ids (Optional[List[str]]): list of node_ids to filter by
            output_fields (Optional[List[str]]): list of fields to return
            embedding_field (Optional[str]): name of embedding field
        """
        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise ValueError(f"Milvus does not support {query.mode} yet.")

        expr = []
        output_fields = ["*"]

        # Parse the filter
        if query.filters is not None:
            expr.extend(_to_milvus_filter(query.filters))

        # Parse any docs we are filtering on
        if query.doc_ids is not None and len(query.doc_ids) != 0:
            expr_list = ['"' + entry + '"' for entry in query.doc_ids]
            expr.append(f"{self.doc_id_field} in [{','.join(expr_list)}]")

        # Parse any nodes we are filtering on
        if query.node_ids is not None and len(query.node_ids) != 0:
            expr_list = ['"' + entry + '"' for entry in query.node_ids]
            expr.append(f"{MILVUS_ID_FIELD} in [{','.join(expr_list)}]")

        # Limit output fields
        if query.output_fields is not None:
            output_fields = query.output_fields

        # Convert to string expression
        string_expr = ""
        if len(expr) != 0:
            string_expr = " and ".join(expr)

        search_params = {
            "metric_type": self.similarity_metric,
            "params": {
                "radius": self.radius,
                "range_filter": 1.0
            }
        }
        # Perform the search
        res = self.milvusclient.search(
            collection_name=self.collection_name,
            data=[query.query_embedding],
            filter=string_expr,
            limit=query.similarity_top_k,
            output_fields=output_fields,
            search_params=search_params,
        )

        logger.debug(
            f"Successfully searched embedding in collection: {self.collection_name}"
            f" Num Results: {len(res[0])}"
        )

        nodes = []
        similarities = []
        ids = []

        # print('................milvus query.....')
        redis = RedisUtils(config.REDIS_DB)
        # Parse the results
        for hit in res[0]:
            if not self.text_key:
                node = metadata_dict_to_node(
                    {"_node_content": hit["entity"].get("_node_content", None)}
                )
            else:
                try:
                    text = hit["entity"].get(self.text_key)
                except Exception:
                    raise ValueError(
                        "The passed in text_key value does not exist "
                        "in the retrieved entity."
                    )
                node = TextNode(
                    text=text,
                )
            if self.is_qa:
                redis_key = 'qa-' + self.collection_name + '-' + node.node_id
                redis_value = redis.getObject(redis_key, True)
                # print('........redis_value:', redis_value)
                node.text = "Question: " + node.text + "\nFactual answer:" + redis_value
            nodes.append(node)
            similarities.append(hit["distance"])
            ids.append(hit["id"])
        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)