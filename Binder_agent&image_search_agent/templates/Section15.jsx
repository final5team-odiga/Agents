import React from "react";
import styled from "styled-components";

const Section15Wrapper = styled.div`
  align-self: stretch;
  height: 800px;
  max-width: 1000px;
  flex-direction: column;
  justify-content: flex-start;
  align-items: center;
  gap: 321px;
  display: inline-flex;
`;

const CategoryText = styled.div`
  text-align: center;
  color: black;
  font-size: 11px;
  font-family: "Poppins", sans-serif;
  font-weight: 300;
  line-height: 15.4px;
  letter-spacing: 1.1px;
`;

const Title = styled.h1`
  text-align: center;
  color: black;
  font-size: 75px;
  font-family: "Spectral", serif;
  font-weight: 400;
  line-height: 81px;
  margin: 0;
`;

const Subtitle = styled.p`
  text-align: center;
  color: black;
  font-size: 18px;
  font-family: "Spectral", serif;
  font-weight: 400;
  line-height: 28px;
  margin: 0;
`;

export const Section15 = ({ tagline, title, subtitle }) => {
  return (
    <Section15Wrapper>
      <CategoryText>{tagline}</CategoryText>
      <Title>{title}</Title>
      <Subtitle>{subtitle}</Subtitle>
    </Section15Wrapper>
  );
};
