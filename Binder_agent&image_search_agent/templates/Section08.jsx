import React from "react";
import styled from "styled-components";

const StyledSection08 = styled.div`
  align-self: stretch;
  height: 800px;
  max-width: 1000px;
  padding: 50px 48px 95px 48px;
  display: inline-flex;
  flex-direction: column;
  justify-content: flex-start;
  align-items: center;
  gap: 48px;
`;

const StyledTitle = styled.div`
  align-self: stretch;
  text-align: center;
  color: black;
  font-size: 18px;
  font-family: "Spectral", serif;
  font-weight: 400;
  line-height: 28px;
`;

const StyledGrid = styled.div`
  align-self: stretch;
  display: inline-flex;
  justify-content: flex-start;
  align-items: flex-start;
  gap: 20px;
`;

const StoryCard = styled.div`
  flex: 1 1 0;
  display: inline-flex;
  flex-direction: column;
  gap: 12px;
`;

const StoryImage = styled.img`
  align-self: stretch;
  height: ${({ height }) => height || "301px"};
`;

const StoryMeta = styled.div`
  display: flex;
  flex-direction: column;
  gap: 16px;
  align-self: stretch;
`;

const MetaCategory = styled.div`
  color: black;
  font-size: 11px;
  font-family: "Poppins", sans-serif;
  font-weight: 300;
  line-height: 15.4px;
  letter-spacing: 1.1px;
`;

const MetaTextGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
  align-self: stretch;
`;

const MetaTitle = styled.div`
  color: black;
  font-size: 21px;
  font-family: "Spectral", serif;
  font-weight: 400;
  line-height: 27px;
`;

const MetaDescription = styled.div`
  color: #4d4d4d;
  font-size: 16px;
  font-family: "Spectral", serif;
  font-weight: 400;
  line-height: 24px;
`;

export const Section08 = ({ title, body }) => {
  return (
    <StyledSection08>
      <StyledTitle>{title}</StyledTitle>
      <StyledGrid>
        {body.map((story, i) => (
          <StoryCard key={i}>
            <StoryImage
              src={story.imageUrl}
              style={story.imageHeight ? { height: story.imageHeight } : {}}
              alt={story.title}
            />
            <StoryMeta>
              <MetaCategory>{story.category}</MetaCategory>
              <MetaTextGroup>
                <MetaTitle>{story.title}</MetaTitle>
                <MetaDescription>{story.description}</MetaDescription>
              </MetaTextGroup>
            </StoryMeta>
          </StoryCard>
        ))}
      </StyledGrid>
    </StyledSection08>
  );
};
