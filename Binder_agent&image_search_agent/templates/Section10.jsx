import React from "react";
import styled from "styled-components";

const StyledSection10 = styled.div`
  align-self: stretch;
  height: 800px;
  max-width: 1000px;
  padding: 48px;
  flex-direction: column;
  justify-content: flex-start;
  align-items: flex-start;
  gap: 40px;
  display: inline-flex;
`;

const Row = styled.div`
  align-self: stretch;
  display: inline-flex;
  justify-content: flex-start;
  align-items: flex-start;
  gap: 40px;
`;

const ProductBox = styled.div`
  flex: 1 1 0;
  flex-direction: column;
  justify-content: flex-start;
  align-items: flex-start;
  gap: 10px;
  display: inline-flex;
`;

const ProductImage = styled.img`
  align-self: stretch;
  height: 220px;
`;

const ProductTextBlock = styled.div`
  align-self: stretch;
  flex-direction: column;
  justify-content: flex-start;
  align-items: flex-start;
  gap: 22px;
  display: flex;
`;

const ProductLabel = styled.span`
  color: black;
  font-size: 11px;
  font-family: Poppins;
  font-weight: 300;
  line-height: 15.4px;
  letter-spacing: 1.1px;
  word-wrap: break-word;
`;

const ProductPrice = styled.span`
  color: black;
  font-size: 16px;
  font-family: Spectral;
  font-weight: 400;
  line-height: 24px;
  word-wrap: break-word;
`;

export const Section10 = ({ body }) => {
  return (
    <StyledSection10>
      <Row>
        {body.slice(0, 4).map((item, i) => (
          <ProductBox key={i}>
            <ProductImage src={item.imageUrl} alt={`Product ${i + 1}`} />
            <ProductTextBlock>
              <ProductLabel>{item.title}</ProductLabel>
              <ProductPrice>{item.tagline}</ProductPrice>
            </ProductTextBlock>
          </ProductBox>
        ))}
      </Row>
      <Row>
        {body.slice(4, 8).map((item, i) => (
          <ProductBox key={i + 4}>
            <ProductImage src={item.imageUrl} alt={`Product ${i + 5}`} />
            <ProductTextBlock>
              <ProductLabel>{item.title}</ProductLabel>
              <ProductPrice>{item.tagline}</ProductPrice>
            </ProductTextBlock>
          </ProductBox>
        ))}
      </Row>
    </StyledSection10>
  );
};
