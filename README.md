# Formação Engenharia de Dados & Machine Learning no Databricks 🚀

Este repositório contém o projeto prático desenvolvido ao longo do curso, demonstrando a construção de um pipeline de dados completo de ponta a ponta utilizando a **Arquitetura Medallion** no **Databricks Lakehouse**, integrado à governança de dados do **Unity Catalog** e finalizando com um modelo preditivo.

## Objetivo do Projeto
O cenário simula uma plataforma de e-commerce que rastreia o comportamento de navegação dos usuários. O objetivo final é construir um pipeline de dados robusto capaz de alimentar um modelo de classificação (**Random Forest**) para prever a **propensão de compra de produtos da categoria de Eletrônicos**

## Tecnologias e Recursos Utilizados
* **Databricks Runtime** para computação distribuída Spark.
* **PySpark & Spark SQL** como motores principais de transformação de dados.
* **Delta Lake** para garantia de transações ACID e recursos de auditoria/segurança como *Time Travel* (`RESTORE TABLE`).
* **Unity Catalog** para governança de 3 níveis (`catálogo.esquema.tabela`).
* **Git Folders (Databricks Repos)** para controle de versão integrado diretamente ao GitHub.

---

---
*Este projeto foi desenvolvido como material didático para capacitação em Engenharia de Machine Learning*
